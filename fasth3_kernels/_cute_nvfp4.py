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
