"""SM120 inference epilogues for the H3 attention path."""
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cuda.bindings import driver as cuda
import torch


@cute.kernel
def _layout(x: cute.Tensor, source: cute.Tensor, weight, cos, sin, out: cute.Tensor,
            H: cutlass.Constexpr, P: cutlass.Constexpr, R: cutlass.Constexpr,
            EPS: cutlass.Constexpr, INPUT_HEADS: cutlass.Constexpr, HEAD_OFFSET: cutlass.Constexpr):
    tx, _, _ = cute.arch.thread_idx()
    bx, _, _ = cute.arch.block_idx()
    lane = tx % 32
    task = bx * 4 + tx // 32
    if task < H * P:
        head, padded = task // P, task % P
        token = source[padded].to(cutlass.Int32)
        base = (token * INPUT_HEADS + HEAD_OFFSET + head) * 128
        inv = cutlass.Float32(1.0)
        if cutlass.const_expr(weight is not None):
            if token >= 0:
                squares = cutlass.Float32(0.0)
                for i in cutlass.range_constexpr(4):
                    value = x[base + i * 32 + lane].to(cutlass.Float32)
                    squares += value * value
                for shift in cutlass.range_constexpr(5):
                    squares += cute.arch.shuffle_sync_bfly(squares, 1 << shift)
                inv = cute.math.rsqrt(squares * (1.0 / 128.0) + EPS)
        for i in cutlass.range_constexpr(4):
            col = i * 32 + lane
            value = cutlass.BFloat16(0.0)
            if token >= 0:
                value = x[base + col]
                if cutlass.const_expr(weight is not None):
                    value = (value.to(cutlass.Float32) * inv * weight[col].to(cutlass.Float32)).to(cutlass.BFloat16)
                if cutlass.const_expr(R > 0):
                    if col < R:
                        other_col = (col + R // 2) % R
                        other = x[base + other_col].to(cutlass.Float32)
                        other = (other * inv * weight[other_col].to(cutlass.Float32)).to(cutlass.BFloat16)
                        if col < R // 2:
                            other = -other
                        c = cos[token * R + col].to(cutlass.BFloat16).to(cutlass.Float32)
                        s = sin[token * R + col].to(cutlass.BFloat16).to(cutlass.Float32)
                        left = (value.to(cutlass.Float32) * c).to(cutlass.BFloat16).bitcast(cutlass.Uint16)
                        right = (other.to(cutlass.Float32) * s).to(cutlass.BFloat16).bitcast(cutlass.Uint16)
                        left_fp32 = (left.to(cutlass.Uint32) << 16).bitcast(cutlass.Float32)
                        right_fp32 = (right.to(cutlass.Uint32) << 16).bitcast(cutlass.Float32)
                        value = (left_fp32 + right_fp32).to(cutlass.BFloat16)
            out[task * 128 + col] = value


@cute.jit
def _run_layout(x: cute.Tensor, source: cute.Tensor, weight, cos, sin, out: cute.Tensor,
                H: cutlass.Constexpr, P: cutlass.Constexpr, R: cutlass.Constexpr,
                EPS: cutlass.Constexpr, INPUT_HEADS: cutlass.Constexpr, HEAD_OFFSET: cutlass.Constexpr,
                stream: cuda.CUstream):
    _layout(x, source, weight, cos, sin, out, H, P, R, EPS, INPUT_HEADS, HEAD_OFFSET).launch(
        grid=((H * P + 3) // 4, 1, 1), block=(128, 1, 1), stream=stream)


@cute.kernel
def _compress(fine: cute.Tensor, coarse: cute.Tensor, gate: cute.Tensor,
              inverse: cute.Tensor, out: cute.Tensor, N: cutlass.Constexpr,
              H: cutlass.Constexpr, P: cutlass.Constexpr, TILE: cutlass.Constexpr):
    tx, _, _ = cute.arch.thread_idx()
    bx, _, _ = cute.arch.block_idx()
    index = bx * 256 + tx
    if index < N * H * 128:
        token, head, col = index // (H * 128), index // 128 % H, index % 128
        padded = inverse[token].to(cutlass.Int32)
        a = fine[(head * P + padded) * 128 + col].to(cutlass.Float32)
        b = coarse[(head * (P // TILE) + padded // TILE) * 128 + col].to(out.element_type)
        g = gate[index].to(cutlass.Float32)
        out[index] = cute.math.fma(b.to(cutlass.Float32), g, a).to(out.element_type)


@cute.jit
def _run_compress(fine: cute.Tensor, coarse: cute.Tensor, gate: cute.Tensor,
                  inverse: cute.Tensor, out: cute.Tensor, N: cutlass.Constexpr,
                  H: cutlass.Constexpr, P: cutlass.Constexpr, TILE: cutlass.Constexpr, stream: cuda.CUstream):
    _compress(fine, coarse, gate, inverse, out, N, H, P, TILE).launch(
        grid=((N * H * 128 + 255) // 256, 1, 1), block=(256, 1, 1), stream=stream)


_LAYOUT_CACHE = {}
_COMPRESS_CACHE = {}


def _view(tensor):
    if tensor is not None:
        return from_dlpack(tensor.detach().view(-1), assumed_align=tensor.element_size())


def attention_layout(x, source, norm=None, rotary=None, *, head_slice=None):
    """Write padded [1,H,P,128] directly from token-major BF16 Q, K or V."""
    _, _, input_heads, dim = x.shape
    offset, heads = (0, input_heads) if head_slice is None else head_slice
    if not 0 <= offset < offset + heads <= input_heads:
        raise ValueError('Invalid input head slice')
    padded = source.numel()
    result = x.new_empty((1, heads, padded, dim))
    cos, sin = (tuple(t.to(x.dtype).contiguous() for t in rotary)
                if rotary is not None else (None, None))
    views = tuple(map(_view, (x, source, norm.weight if norm is not None else None, cos, sin, result)))
    stream = cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
    params = (heads, padded, cos.shape[-1] if cos is not None else 0,
              norm.eps if norm is not None else 0.0, input_heads, offset)
    key = (x.device.index, x.numel(), *(t.element_type if t is not None else None for t in views), *params)
    compiled = _LAYOUT_CACHE.get(key)
    if compiled is None:
        compiled = _LAYOUT_CACHE[key] = cute.compile(_run_layout, *views, *params, stream)
    compiled(*views, stream)
    return result


def compress_gate(fine, coarse, gate, inverse, *, tile_size=128):
    """Apply raw compression gates while writing unpadded token-major output."""
    _, heads, padded, dim = fine.shape
    if tile_size not in (64, 128) or padded % tile_size or coarse.shape != (1, heads, padded // tile_size, dim):
        raise ValueError('Compression gate coarse shape differs from the logical tile layout')
    tokens = inverse.numel()
    result = gate.new_empty((1, tokens, heads, dim), dtype=fine.dtype)
    views = tuple(map(_view, (fine, coarse, gate, inverse, result)))
    stream = cuda.CUstream(torch.cuda.current_stream(fine.device).cuda_stream)
    params = tokens, heads, padded, tile_size
    key = (fine.device.index, fine.dtype, coarse.dtype, gate.dtype, *params)
    compiled = _COMPRESS_CACHE.get(key)
    if compiled is None:
        compiled = _COMPRESS_CACHE[key] = cute.compile(_run_compress, *views, *params, stream)
    compiled(*views, stream)
    return result
