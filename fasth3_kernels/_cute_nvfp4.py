"""Independent scalar-parallel CuTe E2M1 decode, not a vendor wrapper."""
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cuda.bindings import driver as cuda


@cute.kernel
def _decode(P: cute.Tensor, S: cute.Tensor, G: cute.Tensor, O: cute.Tensor,
            COUNT: cutlass.Constexpr, K: cutlass.Constexpr):
    tx, _, _ = cute.arch.thread_idx()
    bx, _, _ = cute.arch.block_idx()
    i = bx * 256 + tx
    if i < COUNT:
        packed = P[i // 2].to(cutlass.Int32)
        code = (packed >> ((i % 2) * 4)) & 15
        mag = code & 7
        val = cutlass.Float32(0.0)
        # Simple exact E2M1 arithmetic: exponent e=mag>>1, mantissa m=mag&1.
        # e=0 => m/2; e>=1 => (1+m/2)*2**(e-1).
        if mag == 1:
            val = 0.5
        if mag >= 2:
            exponent = (mag >> 1) - 1
            val = (1.0 + (mag & 1).to(cutlass.Float32) * 0.5) * (1 << exponent).to(cutlass.Float32)
        if (code & 8) != 0:
            val = -val
        sf = S[(i // K) * (K // 16) + (i % K) // 16]
        O[i] = (val * sf * G[0]).to(O.element_type)


@cute.jit
def _run(P: cute.Tensor, S: cute.Tensor, G: cute.Tensor, O: cute.Tensor,
         COUNT: cutlass.Constexpr, K: cutlass.Constexpr, stream: cuda.CUstream):
    _decode(P, S, G, O, COUNT, K).launch(
        grid=((COUNT + 255) // 256, 1, 1), block=(256, 1, 1), stream=stream)


_CACHE = {}


def launch(packed, scales, gs, dtype):
    n, pk = packed.shape
    out = torch.empty((n, pk * 2), device=packed.device, dtype=dtype)
    views = [from_dlpack(t.detach().view(-1), assumed_align=t.element_size())
             for t in (packed, scales, gs, out)]
    stream = cuda.CUstream(torch.cuda.current_stream(packed.device).cuda_stream)
    key = (packed.device.index, n, pk, dtype)
    compiled = _CACHE.get(key)
    if compiled is None:
        compiled = cute.compile(_run, *views, n * pk * 2, pk * 2, stream)
        _CACHE[key] = compiled
    compiled(*views, stream)
    return out


@cute.jit
def _e2m1(code):
    # The nibble-indexed table stores twice each nonnegative E2M1 value.
    bits = (cutlass.Uint32(0xC8643210) >> ((code & 7) * 4)) & 15
    magnitude = bits.to(cutlass.Float32) * 0.5
    return (magnitude.bitcast(cutlass.Uint32) | ((code & 8).to(cutlass.Uint32) << 28)).bitcast(cutlass.Float32)


@cute.kernel
def _decode_packed(P: cute.Tensor, S: cute.Tensor, G: cute.Tensor, O: cute.Tensor,
                   COUNT: cutlass.Constexpr, PK: cutlass.Constexpr, SCALE_TILES: cutlass.Constexpr):
    tx, _, _ = cute.arch.thread_idx()
    bx, _, _ = cute.arch.block_idx()
    i = bx * 256 + tx
    if i < COUNT:
        row = i // PK
        col = i % PK // 8
        # Checkpoint axes are [128-row tile, 4-scale tile, row % 32, row // 32, scale % 4].
        scale = (((row // 128 * SCALE_TILES + col // 4) * 32 + row % 32) * 4 + row // 32 % 4) * 4 + col % 4
        sf = S[scale].to(cutlass.Float32)
        packed = P[i].to(cutlass.Int32)
        low = (_e2m1(packed & 15) * sf * G[0]).to(cutlass.BFloat16).bitcast(cutlass.Uint16)
        high = (_e2m1(packed >> 4) * sf * G[0]).to(cutlass.BFloat16).bitcast(cutlass.Uint16)
        O[i] = low.to(cutlass.Uint32) | (high.to(cutlass.Uint32) << 16)


@cute.jit
def _run_packed(P: cute.Tensor, S: cute.Tensor, G: cute.Tensor, O: cute.Tensor,
                COUNT: cutlass.Constexpr, PK: cutlass.Constexpr, SCALE_TILES: cutlass.Constexpr,
                stream: cuda.CUstream):
    _decode_packed(P, S, G, O, COUNT, PK, SCALE_TILES).launch(
        grid=((COUNT + 255) // 256, 1, 1), block=(256, 1, 1), stream=stream)


_PACKED_CACHE = {}


def launch_packed(packed, scales, gs):
    n, pk = packed.shape
    out = torch.empty((n, pk * 2), device=packed.device, dtype=torch.bfloat16)
    views = [from_dlpack(t.detach().view(-1), assumed_align=t.element_size())
             for t in (packed, scales, gs, out.view(torch.uint32))]
    stream = cuda.CUstream(torch.cuda.current_stream(packed.device).cuda_stream)
    key = (packed.device.index, n, pk, scales.dtype)
    compiled = _PACKED_CACHE.get(key)
    if compiled is None:
        compiled = cute.compile(_run_packed, *views, n * pk, pk, (pk + 31) // 32, stream)
        _PACKED_CACHE[key] = compiled
    compiled(*views, stream)
    return out
