"""CuTe fusion for BF16 H3 encoder convolution inputs."""
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import cuda.bindings.driver as cuda
import torch


@cute.jit
def reflect(x, limit: cutlass.Constexpr):
    r = x
    if x < 0:
        r = -x
    elif x >= limit:
        r = 2 * limit - x - 2
    return r


@cute.jit
def load(A, point, c, T: cutlass.Constexpr, H: cutlass.Constexpr, W: cutlass.Constexpr,
         C: cutlass.Constexpr, P: cutlass.Constexpr, PH: cutlass.Constexpr, PW: cutlass.Constexpr,
         SC: cutlass.Constexpr, ST: cutlass.Constexpr, SH: cutlass.Constexpr, SW: cutlass.Constexpr,
         SILU: cutlass.Constexpr):
    result = cutlass.BFloat16(0)
    t = point // (PH * PW) - 2
    if point < (T + 2) * PH * PW and t >= 0 and c < C:
        y = reflect(point // PW % PH - P, H)
        x = reflect(point % PW - P, W)
        v = A[c * SC + t * ST + y * SH + x * SW]
        if cutlass.const_expr(SILU):
            a = v.to(cutlass.Float32)
            v = a / (cutlass.Float32(1) + cute.math.exp(-a, fastmath=False))
        result = v.to(cutlass.BFloat16)
    return result


@cute.kernel
def padded(A: cute.Tensor, B: cute.Tensor, T: cutlass.Constexpr, H: cutlass.Constexpr,
           W: cutlass.Constexpr, C: cutlass.Constexpr, P: cutlass.Constexpr,
           PH: cutlass.Constexpr, PW: cutlass.Constexpr, SC: cutlass.Constexpr,
           ST: cutlass.Constexpr, SH: cutlass.Constexpr, SW: cutlass.Constexpr,
           SILU: cutlass.Constexpr):
    tx, _, _ = cute.arch.thread_idx()
    block_p, block_c, _ = cute.arch.block_idx()
    num_p = (T + 2) * PH * PW
    c = block_c * 32 + tx % 32
    if cutlass.const_expr(SC == 1):
        # NDHWC input and output can both use channel-contiguous warps.
        for i in cutlass.range_constexpr(4):
            q = block_p * 32 + tx // 32 + i * 8
            if q < num_p and c < C:
                B[q * C + c] = load(A, q, c, T, H, W, C, P, PH, PW, SC, ST, SH, SW, SILU)
    else:
        # NCTHW and NTCHW layouts both have contiguous pixels within a frame.
        p = block_p * 32 + tx % 32
        shared = cutlass.memory.SmemAllocator()
        tile = shared.allocate_tensor(cutlass.BFloat16, cute.make_layout((32, 32), stride=(33, 1)))
        for i in cutlass.range_constexpr(4):
            lane_c = tx // 32 + i * 8
            tile[lane_c, tx % 32] = load(A, p, block_c * 32 + lane_c, T, H, W, C,
                                         P, PH, PW, SC, ST, SH, SW, SILU)
        cute.arch.sync_threads()
        for i in cutlass.range_constexpr(4):
            pixel = tx // 32 + i * 8
            q = block_p * 32 + pixel
            if q < num_p and c < C:
                B[q * C + c] = tile[tx % 32, pixel]


@cute.jit
def run(A, B, T: cutlass.Constexpr, H: cutlass.Constexpr, W: cutlass.Constexpr,
        C: cutlass.Constexpr, P: cutlass.Constexpr, PH: cutlass.Constexpr, PW: cutlass.Constexpr,
        SC: cutlass.Constexpr, ST: cutlass.Constexpr, SH: cutlass.Constexpr, SW: cutlass.Constexpr,
        SILU: cutlass.Constexpr, stream: cuda.CUstream):
    padded(A, B, T, H, W, C, P, PH, PW, SC, ST, SH, SW, SILU).launch(
        grid=(((T+2)*PH*PW+31)//32, (C+31)//32, 1), block=(256, 1, 1), stream=stream)


_CACHE = {}


def prepare_conv_input(x, padding, *, silu):
    _, c, t, h, w = x.shape
    ph, pw = h + padding + 1, w + padding + 1
    out = torch.empty((1, t+2, ph, pw, c), device=x.device, dtype=torch.bfloat16)
    conf = (t, h, w, c, padding, ph, pw, *x.stride()[1:], silu)
    va = from_dlpack(x.as_strided((x.numel(),), (1,)), assumed_align=16)
    vb = from_dlpack(out.view(-1), assumed_align=16)
    stream = cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
    key = (x.device.index, conf, x.dtype)
    compiled = _CACHE.get(key)
    if compiled is None:
        compiled = _CACHE[key] = cute.compile(run, va, vb, *conf, stream)
    compiled(va, vb, stream)
    return out.permute(0, 4, 1, 2, 3)
