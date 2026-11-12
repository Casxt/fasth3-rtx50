"""Strict FastH3 extensions of the pinned official Diffusers H3 transformer."""
import json
from pathlib import Path
import torch
from .attention import add_compression_gates, H3VSAProcessor
from .pdd import PDDProjection


OMNI_PROFILE = 'omni-pdd8'
T2VA_PROFILE = 't2va-v2'


def build_meta_transformer(config, *, profile=OMNI_PROFILE):
    """Shared model geometry for bounded loading and offline weight export."""
    from accelerate import init_empty_weights
    from diffusers import MiniMaxH3Transformer3DModel

    if profile not in (OMNI_PROFILE, T2VA_PROFILE):
        raise ValueError(f'Unsupported FastH3 transformer profile: {profile}')
    config = dict(config)
    pdd_steps = config.pop('pdd_steps', None)
    if (profile == OMNI_PROFILE and pdd_steps not in (None, 32)) or (profile == T2VA_PROFILE and pdd_steps is not None):
        raise ValueError(f'FastH3 projection contract mismatch: {profile}')
    with init_empty_weights(include_buffers=False):
        model = MiniMaxH3Transformer3DModel.from_config(config)
        add_compression_gates(model)
        if profile == OMNI_PROFILE:
            for name in ('proj_out', 'audio_proj_out'):
                original = getattr(model, name)
                setattr(model, name, PDDProjection(original.in_features, original.out_features,
                                                   grid_size=32, bias=original.bias is not None, device='meta'))
    return model.eval().requires_grad_(False)


def load_transformer(directory, device, guard=lambda phase: None, *, profile=OMNI_PROFILE,
                     debug_checks=False, adaln_table=None):
    from fasth3_kernels.mxfp8 import PACKED_SCALE_LAYOUT

    directory = Path(directory)
    manifest = directory / 'mxfp8_manifest.json'
    if not manifest.is_file():
        raise ValueError('DiT requires an exported packed MXFP8 checkpoint')
    if manifest.stat().st_size > (8 << 20):
        raise ValueError('Oversized DiT manifest')
    metadata = json.loads(manifest.read_text(encoding='utf-8'))
    if not isinstance(metadata, dict) or metadata.get('scale_layout') != PACKED_SCALE_LAYOUT:
        raise ValueError('DiT requires a packed MXFP8 scale layout')
    if torch.device(device).type != 'cuda' or torch.cuda.get_device_capability(device) != (12, 0):
        raise ValueError('Packed MXFP8 DiT requires CUDA SM120')
    if metadata.get('profile', OMNI_PROFILE) != profile:
        raise ValueError('Packed MXFP8 checkpoint profile differs from the requested task')
    if profile == T2VA_PROFILE and adaln_table is not None:
        raise ValueError('The PDD AdaLN table is incompatible with the V2 DMD2 schedule')
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    model = build_meta_transformer(config, profile=profile)
    from .storage import PersistentTensorStore
    from .streaming import ManagedStream
    from .offload import BlockStreamedWeights, dit_groups
    from .mxfp8_linear import replace_mxfp8_linears

    store = PersistentTensorStore(directory, backend='file_readinto')
    try:
        replace_mxfp8_linears(model, store, debug_checks=debug_checks)
        if adaln_table is not None:
            from .adaln import install_adaln_table
            install_adaln_table(model, store, adaln_table)
        groups = dit_groups(model)
        stream = BlockStreamedWeights(model, store, device, guard, groups=groups,
                                      cpu_readahead=True, max_pinned_bytes=512 << 20,
                                      fixed_arena_groups=tuple(group.name for group in groups
                                                               if group.name.startswith('block-')))
        model._dit_weight_arena_bytes = stream.fixed_arena_capacity
        handle = ManagedStream(stream, store, 'readahead')
    except BaseException:
        store.close()
        raise
    # Device discovery only, not an Accelerate weight-loading hook.
    from types import SimpleNamespace
    model._hf_hook = SimpleNamespace(execution_device=torch.device(device))
    return model, handle


def _streamed_fused_ffn(ff, hidden_states):
    if not hidden_states.is_cuda or hidden_states.dtype != torch.bfloat16:
        for layer in ff.net:
            hidden_states = layer(hidden_states)
        return hidden_states
    up, down = ff.net[0].proj, ff.net[2]
    flat = hidden_states.flatten(0, -2)
    result = flat.new_empty((len(flat), down.out_features))
    for begin in range(0, len(flat), down.chunk_tokens):
        part = slice(begin, begin + down.chunk_tokens)
        down(up(flat[part]), swiglu=True, output=result[part])
    return result.reshape(*hidden_states.shape[:-1], down.out_features)


def _pipelined_mxfp8_ffn(ff, hidden_states, *, persistent_full=True, pack=None):
    if not hidden_states.is_cuda or hidden_states.dtype != torch.bfloat16:
        return _streamed_fused_ffn(ff, hidden_states)
    from fasth3_kernels._cute_mxfp8_ffn import Schedule, launch_down, launch_up
    from fasth3_kernels.mxfp8 import MXFP8Tensor, quantize_mxfp8_cute

    up, down = ff.net[0].proj, ff.net[2]
    up._validate_weight_once(hidden_states)
    down._validate_weight_once(hidden_states)
    up_weight = MXFP8Tensor(up.weight, up.weight_scale, up.in_features)
    down_weight = MXFP8Tensor(down.weight, down.weight_scale, down.in_features)
    single_tile = Schedule()
    full_tile = Schedule(ctas_per_sm=2) if persistent_full else single_tile
    flat = hidden_states.flatten(0, -2)
    result = flat.new_empty((len(flat), down.out_features))
    for begin in range(0, len(flat), down.chunk_tokens):
        part = slice(begin, begin + down.chunk_tokens)
        chunk = flat[part]
        activation = None if pack is None else pack(chunk, part)
        if len(chunk) < 384:
            down(up(chunk, packed=activation), swiglu=True, output=result[part])
            del activation
            continue
        up_schedule = full_tile if len(chunk) == 512 else single_tile
        if activation is None:
            activation = quantize_mxfp8_cute(chunk, check_finite=False)
        packed = launch_up(activation, up_weight, up.bias, up_schedule)
        del activation
        launch_down(packed, down_weight, down.bias, result[part], single_tile)
        del packed
    return result.reshape(*hidden_states.shape[:-1], down.out_features)


def _long_attention_workspace(layout, heads_per_group=4):
    """Conservative live head panels, routing temporaries and small CUDA workspaces."""
    padded = layout.source.numel()
    tokens = layout.inverse.numel()
    tiles = len(layout.lengths)
    panel = padded * heads_per_group * 128 * 2
    # Fused token-major QKV remains live until all three padded layouts exist.
    # Routing can hold Q/K/V/fine and up to three FP32 score matrices.
    qkv_layout = 3 * (tokens + padded) * heads_per_group * 128 * 2
    return max(qkv_layout, 4 * panel + 3 * heads_per_group * tiles * tiles * 4) + (320 << 20)


def install_vsa(model, layout, budget, *, sparsity=.9, debug_checks=False, long_scratch=None):
    from types import MethodType
    from .long_video import install_long_blocks
    from .mxfp8_linear import MXFP8Linear

    available = budget.available_bytes()
    tokens = layout.inverse.numel()
    long = tokens > 120_000
    packed = not long and available >= (8 << 30)
    workspace = _long_attention_workspace(layout) if long else 0
    arena = getattr(model, '_dit_weight_arena_bytes', 0)
    activation_bytes = 0
    if long and long_scratch is not None:
        shape = (1, tokens, model.context_embedder.out_features)
        activation_bytes = long_scratch.prepare_cuda(
            shape, layout.inverse.device, budget, output_width=model.transformer_blocks[0].attn.heads * 128,
            workspace_bytes=workspace + arena)
        packed = bool(activation_bytes)
    cache_bytes = (384 << 20) if packed and not long else 0
    tma_store = packed and not long and not debug_checks
    for layer in model.modules():
        if isinstance(layer, MXFP8Linear):
            layer.tma_store = tma_store
    for block in model.transformer_blocks:
        if long:
            for projection in (block.attn.to_q, block.attn.to_k, block.attn.to_v):
                projection.chunk_tokens = 2048
        block.attn.set_processor(H3VSAProcessor(layout, sparsity=sparsity, packed_output=packed,
                                                cache_bytes=cache_bytes,
                                                host_scratch=long_scratch if long or not packed else None,
                                                reuse_input=long or not packed))
        up, down = block.ff.net[0].proj, block.ff.net[2]
        native = (not (up.debug_checks or down.debug_checks)
                  and (up.in_features, up.out_features, down.in_features, down.out_features)
                  == (5376, 28672, 14336, 5376))
        block.ff.forward = MethodType(_pipelined_mxfp8_ffn if native else _streamed_fused_ffn, block.ff)
        block._pack_ffn_adaln = native
    install_long_blocks(model, long_scratch if long or not packed else None)
    policy = {'attention_output': 'mxfp8' if packed else 'host', 'qkv_cache_bytes': cache_bytes,
            'available_bytes': available, 'tokens': tokens,
            'qkv_chunk_tokens': 2048 if long else 512,
            'activation_scratch_reuse': (long or not packed) and long_scratch is not None,
            'activation_arena_bytes': activation_bytes,
            'attention_workspace_bytes': workspace, 'weight_arena_bytes': arena,
            'attention_output_bytes': tokens * (model.transformer_blocks[0].attn.heads * 132) if packed else 0,
            'tma_store': tma_store, 'host_output_reuse': not packed and long_scratch is not None,
            'host_output_inplace': not packed, 'attention_output_inplace': long or not packed}
    if long_scratch is not None:
        long_scratch.policy = policy
    return policy
