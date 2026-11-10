"""Bounded Ref2VA video ingestion and exact head-independent VSA scheduling."""
from math import floor, sqrt
from types import MethodType

import torch
from torch.nn import functional as F


def aligned_video_frames(frames):
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import align_num_frames
    return align_num_frames(frames, 17, 5)


def allow_released_maximum(pipe):
    """Allow FastVideo's last aligned VAE chunk on this Diffusers snapshot."""
    class LongPipeline(type(pipe)):
        @property
        def max_duration(self):
            return aligned_video_frames(15 * self.fps) / self.fps

    pipe.__class__ = LongPipeline


def read_video_reference(path, canvas, frames):
    """Decode directly to the model canvas, retaining only normalized 24 fps frames."""
    import av
    import numpy as np
    from importlib.util import find_spec
    from PIL import Image
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import resolve_canvas_size
    from diffusers.modular_pipelines.minimax_h3.references import MiniMaxH3VideoReference

    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"No video stream in {path}")
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        stream.codec_context.thread_count = 4
        rate = stream.average_rate or stream.guessed_rate
        if not rate or rate <= 0:
            raise ValueError(f"No valid video frame rate in {path}")
        clock = 24.0 / float(rate)
        result = None
        count = 0
        for index, frame in enumerate(container.decode(stream)):
            start = floor(index * clock + .5)
            end = min(frames, floor((index + 1) * clock + .5))
            if end > start:
                pixels = frame.to_ndarray(format="rgb24")
                turns = round(float(frame.rotation or 0) / 90) % 4
                if turns:
                    pixels = np.rot90(pixels, k=-turns)
                if result is None:
                    height, width = resolve_canvas_size(pixels.shape[1], pixels.shape[0], *canvas)
                    result = np.empty((frames, height, width, 3), dtype=np.uint8)
                if pixels.shape[:2] != (height, width):
                    pixels = np.asarray(Image.fromarray(pixels).resize((width, height), Image.Resampling.LANCZOS))
                result[start:end] = pixels
                count = end
            if end >= frames:
                break
        if count == 0:
            raise ValueError(f"No normalized video frames in {path}")
        waveform, sample_rate = None, None
        if container.streams.audio:
            container.seek(0)
            stream = container.streams.audio[0]
            source_rate = int(stream.codec_context.sample_rate)
            # On lean installations avoid making long A/V input depend on optional torchaudio.
            sample_rate = source_rate if find_spec("torchaudio") else 32000
            max_samples = floor(frames * sample_rate / 24)
            resampler = av.audio.resampler.AudioResampler(format="fltp", layout=stream.layout, rate=sample_rate)
            chunks, samples = [], 0
            for frame in container.decode(stream):
                for part in resampler.resample(frame):
                    values = torch.from_numpy(part.to_ndarray())[..., :max_samples - samples]
                    chunks.append(values)
                    samples += values.shape[-1]
                if samples >= max_samples:
                    break
            if samples < max_samples:
                chunks.extend(torch.from_numpy(part.to_ndarray()) for part in resampler.resample(None))
            if chunks:
                waveform = torch.cat(chunks, dim=-1)[..., :max_samples]
    return MiniMaxH3VideoReference(frames=result[:count], fps=24.0, audio=waveform, sample_rate=sample_rate)


def _head_projections(attn, flat, heads, cache=None):
    from fasth3_kernels.mxfp8 import quantize_mxfp8_cute

    layers = (attn.to_q, attn.to_k, attn.to_v)
    first, last = (head * 128 for head in heads)
    width = last - first
    results = tuple(flat.new_empty((len(flat), width)) for _ in layers)
    chunk_tokens = min(layer.chunk_tokens for layer in layers)
    debug = any(layer.debug_checks for layer in layers)
    for begin in range(0, len(flat), chunk_tokens):
        part = slice(begin, begin + chunk_tokens)
        tokens = flat[part]
        packed = cache.get(begin) if cache is not None else None
        if packed is None:
            packed = quantize_mxfp8_cute(tokens, check_finite=debug)
        for layer, output in zip(layers, results):
            layer(tokens, packed=packed, output=output[part], columns=(first, last))
    return [result.view(1, len(flat), width // 128, 128) for result in results]


def _fused_head_projections(attn, flat, first, last, cache):
    from fasth3_kernels.mxfp8 import MXFP8Tensor, quantize_mxfp8_cute
    from .mxfp8_linear import _native_validated_gemm

    layers = (attn.to_q, attn.to_k, attn.to_v)
    lo, hi = first * 128, last * 128
    if (hi - lo != 512 or flat.shape[1] != 5376 or
            not all(layer.packed_scales and layer.chunk_tokens == 512 and not layer.debug_checks and
                    layer.bias is None for layer in layers)):
        return None
    weight = MXFP8Tensor(torch.cat([layer.weight[lo:hi] for layer in layers]),
                         torch.cat([layer.weight_scale[lo // 32:hi // 32] for layer in layers]),
                         flat.shape[1])
    result = flat.new_empty((len(flat), 3 * (hi - lo)))
    for begin in range(0, len(flat), 512):
        part = slice(begin, begin + 512)
        activation = cache.get(begin) if cache is not None else None
        if activation is None:
            activation = quantize_mxfp8_cute(flat[part], check_finite=False)
        _native_validated_gemm(activation, weight, tiled=True, optimized=(64, 128, 3, 2, 2),
                               out=result[part], packed_b=True, tma=True,
                               tma_store=layers[0].tma_store and len(activation.values) % 64 == 0)
    return result.view(1, len(flat), 3 * (last - first), 128)


def _attend_heads(processor, attn, flat, rotary, first, last, cache=None):
    from fasth3_kernels._cute_h3_epilogues import attention_layout, compress_gate
    from fasth3_kernels.attention import sparse_attention
    from .attention import select_block_indices

    layout = processor.layout
    packed = _fused_head_projections(attn, flat, first, last, cache)
    if packed is None:
        projections = _head_projections(attn, flat, (first, last), cache)
    if packed is None:
        q = attention_layout(projections.pop(0), layout.source, attn.norm_q, rotary)
        k = attention_layout(projections.pop(0), layout.source, attn.norm_k, rotary)
        v = attention_layout(projections.pop(0), layout.source)
    else:
        width = last - first
        q = attention_layout(packed, layout.source, attn.norm_q, rotary, head_slice=(0, width))
        k = attention_layout(packed, layout.source, attn.norm_k, rotary, head_slice=(width, width))
        v = attention_layout(packed, layout.source, head_slice=(2 * width, width))
        del packed
    qp, kp, vp = (layout.pool(value.transpose(1, 2)) for value in (q, k, v))
    scores = (qp @ kp.transpose(-2, -1)) / sqrt(128)
    selected = select_block_indices(scores, layout, sparsity=processor.sparsity)
    fine = sparse_attention(q, k, v, selected, tile_lengths=layout.lengths,
                            query_tile_lengths=layout.lengths, dense_prefix=layout.prefix_tiles,
                            tile_size=layout.tile_size)
    del q, k, v, selected, qp, kp
    compressed = scores.softmax(-1) @ vp
    del scores, vp
    lo, hi = first * 128, last * 128
    gate = F.linear(flat, attn.to_gate_compress.weight[lo:hi]).view(1, len(flat), last-first, 128)
    return compress_gate(fine, compressed.contiguous(), gate, layout.inverse,
                         tile_size=layout.tile_size).view(len(flat), hi-lo)


def _cache_qkv_activations(attn, flat, budget):
    from fasth3_kernels.mxfp8 import quantize_mxfp8_cute

    layers = (attn.to_q, attn.to_k, attn.to_v)
    chunk_size = min(layer.chunk_tokens for layer in layers)
    debug = any(layer.debug_checks for layer in layers)
    row_bytes = flat.shape[1] + flat.shape[1] // 32
    cache = {}
    for begin in range(0, len(flat), chunk_size):
        chunk = flat[begin:begin + chunk_size]
        cost = len(chunk) * row_bytes
        if cost > budget:
            break
        cache[begin] = quantize_mxfp8_cute(chunk, check_finite=debug)
        budget -= cost
    return cache


def _project_packed_attention(layer, values, scales, flat):
    from fasth3_kernels.mxfp8 import MXFP8Tensor

    result = flat.new_empty((len(flat), layer.out_features))
    # MXFP8Linear takes the BF16 shape/dtype plus its already-quantized operand.
    # With packed supplied the template is never read by the GEMM.
    template = flat.new_empty((1, layer.in_features))
    for begin in range(0, len(flat), layer.chunk_tokens):
        part = slice(begin, begin + layer.chunk_tokens)
        packed = MXFP8Tensor(values[part], scales[part], layer.in_features)
        layer(template.expand(len(packed.values), -1), packed=packed, output=result[part])
    return result


class _PinnedHeadOutput:
    """Overlap at most two GPU-to-host head-group copies with compute."""

    def __init__(self, host, columns, device):
        self.host = host
        self.copy_stream = torch.cuda.Stream(device=device)
        self.slots = [torch.empty((len(host), columns), dtype=host.dtype, pin_memory=True) for _ in range(2)]
        self.pending = [None, None]

    def _drain(self, slot):
        pending = self.pending[slot]
        if pending is not None:
            event, columns, buffer = pending
            event.synchronize()
            self.host[:, columns].copy_(buffer)
            self.pending[slot] = None

    def copy(self, group, columns, slot):
        self._drain(slot)
        buffer = self.slots[slot][:, :group.shape[1]]
        self.copy_stream.wait_stream(torch.cuda.current_stream(group.device))
        with torch.cuda.stream(self.copy_stream):
            buffer.copy_(group, non_blocking=True)
            group.record_stream(self.copy_stream)
            event = self.copy_stream.record_event()
        self.pending[slot] = event, columns, buffer

    def finish(self):
        self._drain(0)
        self._drain(1)


class LongVideoScratch:
    """Keep the host staging and fixed-shape CUDA buffers for one long request."""

    def __init__(self):
        self.host = self.pinned = None
        self.packed = self.modulated = None

    def prepare_cuda(self, shape, device, budget):
        # Two BF16 matrices: the residual sequence and its disposable AdaLN input.
        required = 4 * shape[0] * shape[1] * shape[2]
        if required > budget.available_bytes():
            return 0
        try:
            self.packed = torch.empty(shape, device=device, dtype=torch.bfloat16)
            self.modulated = torch.empty_like(self.packed)
        except torch.OutOfMemoryError:
            self.packed = self.modulated = None
            torch.cuda.empty_cache()
            return 0
        return required

    def acquire(self, flat, width, columns):
        shape = (len(flat), width)
        if self.host is None:
            self.host = torch.empty(shape, dtype=flat.dtype, device="cpu")
            # Cap the two activation slots separately from weight staging.
            if len(flat) * columns * flat.element_size() * 2 <= 512 << 20:
                self.pinned = _PinnedHeadOutput(self.host, columns, flat.device)
        elif self.host.shape != shape or self.host.dtype != flat.dtype:
            raise ValueError("Host attention scratch requires fixed request geometry")
        return self.host, self.pinned

    def close(self):
        try:
            if self.pinned is not None:
                self.pinned.copy_stream.synchronize()
        finally:
            self.pinned = self.host = None
            self.packed = self.modulated = None


def grouped_attention(processor, attn, hidden_states, rotary_emb, heads_per_group=4):
    """Bound attention workspace by independent heads; keep all tokens and routes."""

    if hidden_states.ndim != 3 or hidden_states.shape[0] != 1 or hidden_states.dtype != torch.bfloat16:
        raise ValueError("Long-video attention requires one packed BF16 sequence")
    flat = hidden_states[0]
    rotary = tuple(value.to(flat.dtype).contiguous() for value in rotary_emb) if rotary_emb is not None else None
    width = attn.heads * 128
    packed_output = processor.packed_output
    cache_budget = processor.cache_bytes
    cache = _cache_qkv_activations(attn, flat, cache_budget) if cache_budget else None
    pinned = None
    if packed_output:
        from fasth3_kernels.mxfp8 import quantize_mxfp8_cute
        values = flat.new_empty((len(flat), width), dtype=torch.float8_e4m3fn)
        scales = flat.new_empty((len(flat), width // 32), dtype=torch.uint8)
    else:
        scratch = processor.host_scratch or LongVideoScratch()
        host, pinned = scratch.acquire(flat, width, heads_per_group * 128)
    try:
        for first in range(0, attn.heads, heads_per_group):
            last = min(first + heads_per_group, attn.heads)
            group = _attend_heads(processor, attn, flat, rotary, first, last, cache)
            columns = slice(first * 128, last * 128)
            if packed_output:
                packed = quantize_mxfp8_cute(group, check_finite=attn.to_out[0].debug_checks)
                values[:, columns].copy_(packed.values)
                scales[:, first * 4:last * 4].copy_(packed.scales)
                del packed
            elif pinned is not None:
                pinned.copy(group, columns, (first // heads_per_group) % 2)
            else:
                host[:, columns].copy_(group.cpu())
            del group
        if pinned is not None:
            pinned.finish()
    finally:
        if pinned is not None:
            pinned.copy_stream.synchronize()
    del cache
    out_layer = attn.to_out[0]
    if packed_output:
        return attn.to_out[1](_project_packed_attention(out_layer, values, scales, flat).unsqueeze(0))
    # All heads have finished reading the disposable modulated block input.
    # The residual lives in a different tensor, so the projection can fill this buffer.
    result = flat if processor.reuse_input else flat.new_empty((len(flat), out_layer.out_features))
    for begin in range(0, len(flat), 2048):
        part = slice(begin, begin + 2048)
        out_layer(host[part].to(flat.device), output=result[part])
    return attn.to_out[1](result.unsqueeze(0))


def _modulate_chunk(states, norm, shift, scale, rows):
    return norm(states) * (1.0 + scale.index_select(0, rows)) + shift.index_select(0, rows)


def _modulate(states, norm, shift, scale, indices, *, output_dtype=None, out=None):
    result = torch.empty_like(states, dtype=output_dtype) if out is None else out
    for begin in range(0, states.shape[1], 2048):
        part = slice(begin, begin + 2048)
        result[:, part] = _modulate_chunk(states[:, part], norm, shift, scale, indices[part])
    return result


def _add_residual(branch, residual, gate, indices):
    for begin in range(0, branch.shape[1], 2048):
        part = slice(begin, begin + 2048)
        residual[:, part].add_(branch[:, part] * gate.index_select(0, indices[part]))
    return residual


def _ffn_residual(block, states, shift, scale, gate, indices):
    fuse_pack = (getattr(block, '_pack_ffn_adaln', False) and states.shape[0] == 1 and states.is_cuda
                 and states.dtype == shift.dtype == scale.dtype == torch.bfloat16)
    if fuse_pack:
        from fasth3_kernels._cute_adaln import pack_adaln
    norm = block.norm2
    fuse_rms = (fuse_pack and type(norm) is torch.nn.RMSNorm and norm.weight is not None
                and norm.weight.dtype == states.dtype and norm.eps is not None)
    # 2048 rows preserves the FFN's 512-token GEMM chunk boundaries.
    for begin in range(0, states.shape[1], 2048):
        part = slice(begin, begin + 2048)
        residual = states[:, part]
        rows = indices[part]
        if fuse_rms:
            output = block.ff(residual, pack=lambda chunk, tile: pack_adaln(
                chunk, shift, scale, rows[tile], rms_weight=norm.weight, eps=norm.eps))
        elif fuse_pack:
            normalized = norm(residual)
            output = block.ff(normalized, pack=lambda chunk, tile: pack_adaln(chunk, shift, scale, rows[tile]))
            del normalized
        else:
            output = block.ff(_modulate_chunk(residual, norm, shift, scale, rows).to(states.dtype))
        residual.add_(output * gate.index_select(0, rows))
        del output
    return states


def _block_forward(block, hidden_states, temb, adaln_indices, rotary_emb, attention_mask=None):
    sa, ca, ga, sf, cf, gf = block.adaln_proj(temb)
    scratch = block._long_scratch
    modulated = _modulate(hidden_states, block.norm1, sa, ca, adaln_indices,
                          out=scratch.modulated if scratch is not None else None)
    attn = block.attn(modulated, rotary_emb, attention_mask)
    del modulated
    _add_residual(attn, hidden_states, ga, adaln_indices)
    del attn
    return _ffn_residual(block, hidden_states, sf, cf, gf, adaln_indices)


def _output_norm_forward(norm_out, hidden_states, temb, timestep_indices):
    shift, scale = norm_out.linear(F.silu(temb).to(norm_out.linear.weight.dtype)).chunk(2, dim=-1)
    return _modulate(hidden_states, norm_out.norm, shift, scale, timestep_indices, output_dtype=torch.float32)


def _project_into(packed, rows, projection, source):
    from diffusers.models.modeling_utils import get_parameter_dtype

    dtype = get_parameter_dtype(projection)
    for begin in range(0, len(rows), 4096):
        part = slice(begin, begin + 4096)
        values = projection(source[:, part].to(dtype))
        packed.index_copy_(1, rows[part], values.to(packed.dtype))


def _project_output_rows(packed, rows, time_indices, norm, temb, projection):
    from diffusers.models.modeling_utils import get_parameter_dtype

    dtype = get_parameter_dtype(projection)
    width = getattr(projection, 'head_features', projection.out_features)
    result = packed.new_empty((packed.shape[0], len(rows), width), dtype=dtype)
    for begin in range(0, len(rows), 8192):
        part = slice(begin, begin + 8192)
        indices = rows[part]
        state = packed.index_select(1, indices)
        normalized = norm(state, temb, time_indices.index_select(0, indices))
        result[:, part] = projection(normalized.to(dtype))
    return result


def _model_forward(model, hidden_states, audio_hidden_states, encoder_hidden_states,
                   timestep, timestep_indices, token_tags, position_ids, video_indices,
                   audio_indices, text_indices, attention_kwargs=None, return_dict=True):
    from diffusers.models.modeling_utils import get_parameter_dtype
    from diffusers.models.transformers.transformer_minimax_h3 import (
        MINIMAX_H3_MODALITY_NUM, MiniMaxH3TransformerOutput,
    )

    length = len(position_ids)
    if position_ids.shape != (length, 3) or token_tags.shape != (length,) or timestep_indices.shape != (length,):
        raise ValueError('Invalid long-video packed sequence geometry')
    if attention_kwargs or torch.is_grad_enabled():
        raise ValueError('Long-video execution requires inference without attention overrides')
    dtype = get_parameter_dtype(model.context_embedder)
    shape = (hidden_states.shape[0], length, model.context_embedder.out_features)
    scratch = model._long_scratch
    if scratch is not None and scratch.packed is not None:
        packed = scratch.packed
        if packed.shape != shape or packed.dtype != dtype or packed.device != hidden_states.device:
            raise ValueError('Long-video CUDA scratch differs from request geometry')
        packed.zero_()
    else:
        packed = hidden_states.new_zeros(shape, dtype=dtype)
    _project_into(packed, video_indices, model.proj_in, hidden_states)
    _project_into(packed, audio_indices, model.audio_proj_in, audio_hidden_states)
    static = getattr(model, '_long_static', None)
    if static is not None and 'text' in static:
        text = static['text']
    else:
        text = model.token_refiner(model.context_embedder(encoder_hidden_states.to(dtype)))
        if static is not None:
            static['text'] = text
    packed.index_copy_(1, text_indices, text)
    del text

    if static is not None and 'rotary' in static:
        rotary = static['rotary']
    else:
        rotary = tuple(x.to(dtype).contiguous() for x in model.rope(position_ids))
        if static is not None:
            static['rotary'] = rotary
    temb = model.time_embedder(model.time_proj(timestep).to(get_parameter_dtype(model.time_embedder)))
    indices = timestep_indices * MINIMAX_H3_MODALITY_NUM + token_tags
    for block in model.transformer_blocks:
        packed = block(packed, temb, indices, rotary)

    video = _project_output_rows(packed, video_indices, timestep_indices, model.norm_out, temb, model.proj_out)
    audio = _project_output_rows(packed, audio_indices, timestep_indices, model.norm_out, temb, model.audio_proj_out)
    return MiniMaxH3TransformerOutput(sample=video, audio_sample=audio) if return_dict else (video, audio)


def install_long_blocks(model, scratch=None):
    model._long_scratch = scratch
    for block in model.transformer_blocks:
        block._long_scratch = scratch
        block.forward = MethodType(_block_forward, block)
    model.norm_out.forward = MethodType(_output_norm_forward, model.norm_out)
    model.forward = MethodType(_model_forward, model)
