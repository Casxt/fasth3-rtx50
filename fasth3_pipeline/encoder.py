"""Bounded Qwen3-VL NVFP4 conditioner, independent of the reference runtime.

The local encoder contains only the first 50 language blocks. H3 requires the
*pre-final-norm* layer-50 state, not the normalized final output of that truncated
stack. We capture that state explicitly; no LM head or missing later blocks are
synthesized. Visual weights are resident only for the visual forward (~1 GiB),
including pos_embed.weight which Transformers accesses without calling Embedding.
Language weights are streamed one block at a time, never a full BF16 state dict.
"""
from pathlib import Path
import json
import types

import torch
from torch import nn
from torch.nn import functional as F

from .storage import PersistentTensorStore


def _qwen_padding_mask(*, attention_mask, **kwargs):
    # H3 supplies an unpadded sequence. Retain a real padding mask as BxK;
    # the attention adapter combines it with the causal mask per query chunk.
    if attention_mask is None or bool(attention_mask.all()):
        return None
    return attention_mask


def bounded_qwen_attention(module, query, key, value, attention_mask, dropout=0., scaling=None, **kwargs):
    """Inference prefill with exact causal visibility and bounded score storage."""
    batch, heads, length, _ = query.shape
    if dropout or length != key.shape[2] or length != value.shape[2]:
        raise ValueError('Bounded Qwen attention requires inference prefill without a KV cache')
    if heads % key.shape[1] or key.shape[1] != value.shape[1]:
        raise ValueError('Invalid Qwen grouped query heads')
    groups = heads // key.shape[1]
    if query.is_cuda and attention_mask is None and groups > 1:
        from torch.backends import cuda
        from torch.nn.attention import SDPBackend, sdpa_kernel

        params = cuda.SDPAParams(query, key, value, None, 0., True, True)
        for backend, supports in ((SDPBackend.FLASH_ATTENTION, cuda.can_use_flash_attention),
                                  (SDPBackend.CUDNN_ATTENTION, cuda.can_use_cudnn_attention)):
            if supports(params):
                with sdpa_kernel(backend):
                    result = F.scaled_dot_product_attention(query, key, value, is_causal=True,
                                                            scale=scaling, enable_gqa=True)
                return result.transpose(1, 2).contiguous(), None

    if groups > 1:
        # The efficient CUDA backend on Windows needs equal head counts.
        key = key.repeat_interleave(groups, dim=1)
        value = value.repeat_interleave(groups, dim=1)

    if query.is_cuda and attention_mask is None:
        from torch.backends import cuda
        from torch.nn.attention import SDPBackend, sdpa_kernel

        params = cuda.SDPAParams(query, key, value, None, 0., True, False)
        backends = [backend for backend, supports in (
            (SDPBackend.FLASH_ATTENTION, cuda.can_use_flash_attention),
            (SDPBackend.EFFICIENT_ATTENTION, cuda.can_use_efficient_attention),
            (SDPBackend.CUDNN_ATTENTION, cuda.can_use_cudnn_attention),
        ) if supports(params)]
        if backends:
            with sdpa_kernel(backends, set_priority=True):
                result = F.scaled_dot_product_attention(query, key, value, is_causal=True, scale=scaling)
            return result.transpose(1, 2).contiguous(), None

    # At most 16M score elements per chunk (~64 MiB in the FP32 math
    # backend). Each query uses one softmax over its complete key prefix.
    chunk = max(1, min(256, (16 << 20) // (batch * heads * length)))
    result = value.new_empty((batch, length, heads, value.shape[-1]))
    positions = torch.arange(length, device=query.device)
    for start in range(0, length, chunk):
        end = min(start + chunk, length)
        causal = positions[:end] <= positions[start:end, None]
        mask = causal
        if attention_mask is not None:
            if attention_mask.ndim == 2:
                mask = causal & attention_mask[:, None, None, :end]
            elif attention_mask.dtype == torch.bool:
                mask = causal & attention_mask[..., start:end, :end]
            else:
                mask = attention_mask[..., start:end, :end].masked_fill(~causal, -torch.inf)
        out = F.scaled_dot_product_attention(query[:, :, start:end], key[:, :, :end], value[:, :, :end],
                                              attn_mask=mask, scale=scaling)
        result[:, start:end].copy_(out.transpose(1, 2))
    return result, None


def _install_bounded_attention(model):
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    name = 'fasth3_bounded_sdpa'
    ALL_ATTENTION_FUNCTIONS.register(name, bounded_qwen_attention)
    ALL_MASK_ATTENTION_FUNCTIONS.register(name, _qwen_padding_mask)
    # The visual encoder already computes independent full-attention blocks.
    model.model.language_model.config._attn_implementation = name


class EncoderTensorStore(PersistentTensorStore):
    index_filename = 'model.safetensors.index.json'


class NVFP4Linear(nn.Module):
    """Original packed checkpoint buffers + explicit BF16 GEMM fallback."""
    def __init__(self, linear, store, prefix, backend='cute'):
        super().__init__()
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.backend = backend
        shapes = {'weight_packed': (torch.uint8, (self.out_features, self.in_features // 2)),
                  'weight_scale': (torch.uint8, None), 'weight_global_scale': (torch.float32, None)}
        if prefix + '.pre_quant_scale' in store.weight_map:
            shapes['pre_quant_scale'] = (torch.bfloat16, (self.in_features,))
        for name, (dtype, expected) in shapes.items():
            shape = tuple(store.shape(prefix + '.' + name))
            if expected is not None and shape != expected:
                raise ValueError(f'Encoder {prefix}.{name} shape {shape} != {expected}')
            self.register_buffer(name, torch.empty(shape, device='meta', dtype=dtype))
        self.bias = linear.bias

    def forward(self, x):
        from fasth3_kernels.nvfp4 import linear_nvfp4
        return linear_nvfp4(x, self.weight_packed, self.weight_scale,
                            self.weight_global_scale, self.bias,
                            pre_quant_scale=getattr(self, 'pre_quant_scale', None),
                            scale_layout='128x4', backend=self.backend, row_chunk=128,
                            global_scale_mode='quantization', decode_slab_bytes=16 << 20)


class _DeviceDiscoveryHook:
    # Device discovery only; real weight loading is handled by our module hooks.
    def __init__(self, device):
        self.execution_device = torch.device(device)

    def pre_forward(self, model, *args, **kwargs):
        return args, kwargs


def _encoder_group_paths(model):
    language = 'model.language_model'
    return ([f'{language}.embed_tokens', 'model.visual']
            + [f'{language}.layers.{i}' for i in range(len(model.model.language_model.layers))]
            + [f'{language}.norm'])


class EncoderStream:
    def __init__(self, model, store, device, guard):
        self.model, self.handles, self.groups = model, [], []
        self.store = store
        try:
            self._initialize(model, store, device, guard)
        except Exception:
            # A format/buffer error after partial hook registration must not
            # leave a half-functional streaming model behind.
            self.close()
            raise

    def _initialize(self, model, store, device, guard):
        store.validate_keys(model)
        self.model, self.device, self.handles = model, torch.device(device), []
        self.groups = []
        # Group visual: HF interpolation directly reads pos_embed.weight. Group
        # language blocks: no access to a sibling's already-evicted parameter.
        covered = set()
        for path in _encoder_group_paths(model):
            module = model.get_submodule(path)
            entries = []
            for child_path, child in module.named_modules():
                for name, value in child._parameters.items():
                    if value is not None:
                        entries.append((child, name, False, value.dtype, f'{path}.{child_path + "." if child_path else ""}{name}'))
                for name, value in child._buffers.items():
                    if value is not None and name not in child._non_persistent_buffers_set:
                        entries.append((child, name, True, value.dtype, f'{path}.{child_path + "." if child_path else ""}{name}'))
            covered.update(e[4] for e in entries)
            self.groups.append(entries)

            def release(mod, args, output, entries=entries):
                for owner, name, buffer, dtype, key in entries:
                    values = owner._buffers if buffer else owner._parameters
                    value = values[name]
                    meta = torch.empty(value.shape, device='meta', dtype=dtype)
                    values[name] = meta if buffer else nn.Parameter(meta, requires_grad=False)

            def load(mod, args, entries=entries, release=release, path=path):
                guard(f'encoder:load:{path}')
                try:
                    for owner, name, buffer, dtype, key in entries:
                        value = store.tensor(key)
                        # Raw E4M3 scale bytes must NOT be numerically cast.
                        if buffer and value.dtype != dtype:
                            raise ValueError(f'Packed encoder tensor dtype mismatch: {key}: {value.dtype} != {dtype}')
                        value = value.to(device=self.device, dtype=dtype)
                        (owner._buffers if buffer else owner._parameters)[name] = value if buffer else nn.Parameter(value, requires_grad=False)
                except Exception:
                    release(mod, args, None)
                    raise

            self.handles.append(module.register_forward_pre_hook(load))
            self.handles.append(module.register_forward_hook(release, always_call=True))
        if covered != set(store.weight_map):
            raise ValueError(f'Uncovered encoder weights: {sorted(set(store.weight_map) - covered)[:8]}')
        for module in model.modules():
            for name in module._non_persistent_buffers_set:
                value = module._buffers.get(name)
                if value is not None:
                    if value.is_meta:
                        raise ValueError(f'Uninitialized nonpersistent encoder buffer: {name}')
                    module._buffers[name] = value.to(self.device)
        model._hf_hook = _DeviceDiscoveryHook(device)

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        for entries in self.groups:
            for owner, name, buffer, dtype, key in entries:
                values = owner._buffers if buffer else owner._parameters
                meta = torch.empty(values[name].shape, device='meta', dtype=dtype)
                values[name] = meta if buffer else nn.Parameter(meta, requires_grad=False)
        if hasattr(self.model, '_hf_hook'):
            del self.model._hf_hook
        close = getattr(self.store, 'close', None)
        if close is not None:
            close()

    def stats_snapshot(self):
        return self.store.stats_snapshot()


def _capture_truncated_hidden_state(model, layer):
    core = model.model
    original = core.forward
    norm = core.language_model.norm

    def forward(self, *args, **kwargs):
        captured = []
        handle = norm.register_forward_pre_hook(lambda mod, inputs: captured.append(inputs[0]))
        kwargs['output_hidden_states'] = False
        kwargs['use_cache'] = False
        kwargs['return_dict'] = True
        try:
            output = original(*args, **kwargs)
        finally:
            handle.remove()
        if len(captured) != 1:
            raise RuntimeError('Expected exactly one final language normalization')
        output.hidden_states = (None,) * layer + (captured[0],)
        return output
    core.forward = types.MethodType(forward, core)


def _prefetch_stream(model, store, device, guard):
    from .offload import BlockStreamedWeights
    from .streaming import ManagedStream

    # The grouped streamer copies serialized bytes. Match the synchronous
    # encoder's compute dtypes and preserve the raw NVFP4 scale buffers.
    for key, value in model.state_dict().items():
        if store.dtype(key) != value.dtype:
            raise ValueError(f'Prefetched encoder dtype mismatch: {key}')
    stream = BlockStreamedWeights(model, store, device, guard, groups=_encoder_group_paths(model),
                                  prefetch_depth=1, guard_each_tensor=False)
    model._hf_hook = _DeviceDiscoveryHook(device)
    managed = ManagedStream(stream, store, 'prefetch')
    managed.add_close_callback(lambda: delattr(model, '_hf_hook'))
    return managed


def load_text_encoder(path, device, guard=lambda phase: None, *, backend='cute', bounded_attention=False,
                      prefetch=False):
    """Return (meta-weight model, closeable stream). No GPU forward is run here.

    ``backend='torch'`` is explicit CPU/GPU diagnostic dequantization only;
    production default is the independently compiled CuTe BF16 fallback.
    """
    from accelerate import init_empty_weights
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    guard('encoder:initialize')
    store = EncoderTensorStore(path, backend='file_readinto' if prefetch else 'safe_open')
    data = json.loads((Path(path) / 'config.json').read_text(encoding='utf-8'))
    quantization = data.pop('quantization_config')
    expected_quantization = {'quant_method': 'nvfp4', 'group_size': 16, 'scale_layout': '128x4',
                             'fmt': 'e2m1', 'scale_fmt': 'e4m3', 'activation_scheme': 'dynamic'}
    if any(quantization.get(key) != value for key, value in expected_quantization.items()):
        raise ValueError('Unsupported encoder NVFP4 checkpoint format')
    kept = data.pop('num_hidden_layers_override')
    original_layers = data['text_config']['num_hidden_layers']
    if kept != 50 or original_layers <= kept:
        raise ValueError('Only the validated 50-layer pre-norm H3 conditioner is supported')
    data['text_config']['num_hidden_layers'] = kept
    config = Qwen3VLConfig.from_dict(data)
    config._attn_implementation = 'sdpa'
    dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        with init_empty_weights(include_buffers=False):
            model = Qwen3VLForConditionalGeneration(config)
    finally:
        torch.set_default_dtype(dtype)
    # This checkpoint is a conditioner, not an autoregressive language model.
    model.lm_head = None
    for key in store.weight_map:
        if not key.endswith('.weight_packed'):
            continue
        prefix = key.removesuffix('.weight_packed')
        parent, _, name = prefix.rpartition('.')
        owner = model.get_submodule(parent)
        linear = getattr(owner, name)
        if not isinstance(linear, nn.Linear):
            raise TypeError(f'NVFP4 checkpoint targets non-linear: {prefix}')
        setattr(owner, name, NVFP4Linear(linear, store, prefix, backend=backend))
    # Keep the advertised original depth so Diffusers accepts layer 50, while
    # the explicit forward adapter returns its captured pre-norm state.
    model.config.text_config.num_hidden_layers = original_layers
    if bounded_attention:
        _install_bounded_attention(model)
    _capture_truncated_hidden_state(model, kept)
    model.eval().requires_grad_(False)
    if prefetch:
        try:
            stream = _prefetch_stream(model, store, device, guard)
        except BaseException:
            store.close()
            raise
    else:
        stream = EncoderStream(model, store, device, guard)
    return model, stream
