"""Compact BF16 Tensor Core attention for H3 tile-64 and tile-128 routing."""
import math
import torch


def sparse_attention(query, key, value, indices, *, tile_lengths, query_tile_lengths, dense_prefix,
                     scale=None, tile_size=128):
    if not isinstance(tile_size, int) or tile_size not in (64, 128):
        raise ValueError('H3 attention requires tile_size 64 or 128')
    if query.ndim != 4 or key.ndim != 4 or key.shape != value.shape:
        raise ValueError('Expected [batch, heads, tokens, head_dim] attention tensors')
    b, h, sq, d = query.shape
    sk = key.shape[-2]
    device = query.device
    if (min(b, h, sq, sk) < 1 or d != 128 or key.shape != (b, h, sk, d)
            or device.type != 'cuda' or torch.cuda.get_device_capability(device) != (12, 0)
            or any(t.device != device or t.dtype != torch.bfloat16 for t in (query, key, value))):
        raise ValueError('H3 attention requires matching SM120 CUDA BF16 tensors with head width 128')
    if torch.is_grad_enabled() and any(t.requires_grad for t in (query, key, value)):
        raise ValueError('CuTe attention is inference-only')
    nq, nk = math.ceil(sq / tile_size), math.ceil(sk / tile_size)
    if (indices.ndim != 4 or indices.shape[:3] != (b, h, nq)
            or indices.dtype != torch.int32 or indices.device != device
            or not 0 <= dense_prefix <= min(nq, nk) or indices.shape[-1] > nk - dense_prefix):
        raise ValueError('Invalid compact sparse metadata')
    for lengths, tiles in ((tile_lengths, nk), (query_tile_lengths, nq)):
        if lengths.shape != (tiles,) or lengths.device != device or lengths.dtype not in (torch.int32, torch.int64):
            raise ValueError('Expected one integer length per logical tile')
    scale = d ** -.5 if scale is None else scale
    if not math.isfinite(scale):
        raise ValueError('Invalid attention scale')
    physical = (sk - torch.arange(nk, device=device) * tile_size).clamp(0, tile_size)
    kl = torch.minimum(tile_lengths.clamp(0, tile_size), physical).int().contiguous()
    ql = query_tile_lengths.clamp(0, tile_size).int().contiguous()
    from ._cute_attention_tc import launch
    return launch(query.contiguous(), key.contiguous(), value.contiguous(), indices.contiguous(),
                  kl, ql, float(scale), dense_prefix, tile_size)
