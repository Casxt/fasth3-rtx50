"""SM120 AdaLN affine and MXFP8 pack for the DiT FFN."""
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.cute.runtime import from_dlpack
from cuda.bindings import driver as cuda
import torch

from ._cute_mxfp8 import _pack_block
from .mxfp8 import MXFP8Tensor


@dsl_user_op
def _affine(x, shift, scale, *, loc=None, ip=None):
    # Explicit .rn prevents ptxas from contracting the eager BF16 multiply/add.
    operands = [t.bitcast(cutlass.Uint16).ir_value(loc=loc, ip=ip) for t in (x, shift, scale)]
    value = llvm.inline_asm(
        cutlass.Uint16.mlir_type, operands,
        "{ .reg .b16 one, temp; mov.b16 one, 0x3f80; add.rn.bf16 temp, $3, one; "
        "mul.rn.bf16 temp, $1, temp; add.rn.bf16 $0, temp, $2; }",
        "=h,h,h,h", loc=loc, ip=ip)
    return cutlass.Uint16(value).to(cutlass.Uint32)


@cute.kernel
def _kernel(X, Shift, Scale, Indices, Codes, Scales,
            M: cutlass.Constexpr, K: cutlass.Constexpr):
    tx, _, _ = cute.arch.thread_idx()
    bx, _, _ = cute.arch.block_idx()
    lane = tx % 32
    block = bx * 4 + tx // 32
    if block < M * (K // 32):
        row, col = block // (K // 32), block % (K // 32) * 32 + lane
        table_row = Indices[row].to(cutlass.Int32)
        bits = _affine(X[row * K + col], Shift[table_row, col], Scale[table_row, col])
        code, exponent = _pack_block(bits)
        Codes[block * 32 + lane] = code
        if lane == 0:
            Scales[block] = exponent


@cute.jit
def _run(X, Shift, Scale, Indices, Codes, Scales,
         M: cutlass.Constexpr, K: cutlass.Constexpr, stream: cuda.CUstream):
    _kernel(X, Shift, Scale, Indices, Codes, Scales, M, K).launch(
        grid=((M * (K // 32) + 3) // 4, 1, 1), block=(128, 1, 1), stream=stream)


@cute.kernel
def _rms_kernel(X, Weight, Shift, Scale, Indices, Codes, Scales,
                K: cutlass.Constexpr, EPS: cutlass.Constexpr):
    tx, _, _ = cute.arch.thread_idx()
    row, _, _ = cute.arch.block_idx()
    lane, warp = tx % 32, tx // 32
    allocator = cutlass.memory.SmemAllocator()
    partials = allocator.allocate_tensor(cutlass.Float32, cute.make_layout(4))
    # Four warps, four consecutive values per thread, and the same tree as
    # PyTorch 2.10's aligned CUDA RMSNorm.
    sumsq = cutlass.Float32(0)
    for i in range((K + 511) // 512):
        col = i * 512 + tx * 4
        if col < K:
            for j in cutlass.range_constexpr(4):
                x = X[row * K + col + j].to(cutlass.Float32)
                sumsq = cute.math.fma(x, x, sumsq)
    for i in cutlass.range_constexpr(5):
        sumsq += cute.arch.shuffle_sync_bfly(sumsq, 16 >> i)
    if lane == 0:
        partials[warp] = sumsq
    cute.arch.sync_threads()
    sumsq = (partials[0] + partials[2]) + (partials[1] + partials[3])
    inverse = cute.math.rsqrt(sumsq / cutlass.Float32(K) + EPS)
    table_row = Indices[row].to(cutlass.Int32)
    for i in range(K // 128):
        col = i * 128 + tx
        x = X[row * K + col].to(cutlass.Float32)
        normalized = (Weight[col].to(cutlass.Float32) * (inverse * x)).to(cutlass.BFloat16)
        bits = _affine(normalized, Shift[table_row, col], Scale[table_row, col])
        code, exponent = _pack_block(bits)
        Codes[row * K + col] = code
        if lane == 0:
            Scales[row * (K // 32) + i * 4 + warp] = exponent


@cute.jit
def _run_rms(X, Weight, Shift, Scale, Indices, Codes, Scales,
             M: cutlass.Constexpr, K: cutlass.Constexpr, EPS: cutlass.Constexpr, stream: cuda.CUstream):
    _rms_kernel(X, Weight, Shift, Scale, Indices, Codes, Scales, K, EPS).launch(
        grid=(M, 1, 1), block=(128, 1, 1), stream=stream)


_CACHE = {}


def pack_adaln(x, shift, scale, rows, *, rms_weight=None, eps=None):
    """Pack row-major MXFP8, optionally normalizing the BF16 input first."""
    m, k = x.shape
    device = x.device
    codes = torch.empty((m, k), device=device, dtype=torch.float8_e4m3fn)
    scales = torch.empty((m, k // 32), device=device, dtype=torch.uint8)
    norm = () if rms_weight is None else (rms_weight,)
    tensors = (x.reshape(-1), *norm, shift, scale, rows, codes.view(-1), scales.view(-1))
    views = [from_dlpack(t.detach(), assumed_align=t.element_size()) for t in tensors]
    stream = cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)
    key = (device.index, m, k, tuple(shift.shape), shift.stride(), scale.stride(), eps)
    compiled = _CACHE.get(key)
    if compiled is None:
        if rms_weight is None:
            compiled = cute.compile(_run, *views, m, k, stream)
        else:
            compiled = cute.compile(_run_rms, *views, m, k, eps, stream)
        _CACHE[key] = compiled
    compiled(*views, stream)
    return MXFP8Tensor(codes, scales, k)
