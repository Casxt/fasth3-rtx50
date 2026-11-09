"""Experimental SM120 native block-scaled warp MMA and BF16 quantization.

The scalar reference uses one warp per 16x8 tile. The tiled variant stages
operands in shared memory and reuses them across four warps.
"""
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cuda.bindings import driver as cuda
from cutlass.cute.nvgpu import warp


@cute.kernel
def _kernel(A: cute.Tensor, B: cute.Tensor, SA: cute.Tensor, SB: cute.Tensor,
            O: cute.Tensor, mma: cute.TiledMma,
            M: cutlass.Constexpr, N: cutlass.Constexpr, K: cutlass.Constexpr):
    lane, _, _ = cute.arch.thread_idx()
    bm, bn, _ = cute.arch.block_idx()
    thr = mma.get_slice(lane)
    ca = thr.partition_A(cute.make_identity_tensor((16, 32)))
    cb = thr.partition_B(cute.make_identity_tensor((8, 32)))
    cc = thr.partition_C(cute.make_identity_tensor((16, 8)))
    # ca/cb/cc are coordinate tensors: their ScaledBasis strides map to
    # (row,col) tuples, not scalar register addresses. Installed DSL 4.8.1
    # make_rmem_tensor_like explicitly handles CoordTensorType by keeping the
    # domain shape and constructing compact scalar strides (tensor.py:1069).
    # Retain ca/cb/cc separately for scalar-load and output coordinate maps.
    ra = cute.make_rmem_tensor_like(ca, A.element_type)
    rb = cute.make_rmem_tensor_like(cb, B.element_type)
    acc = cute.make_rmem_tensor_like(cc, cutlass.Float32)
    acc.fill(0.0)
    # Native scale fragment: one scale per row / K32. Repetition is deliberate
    # and harmless for this probe. SF source-lane mapping is distinct from A/B
    # data-register mapping; varied per-row scales in probe catch mistakes.
    sfa = cute.make_rmem_tensor((32,), cutlass.Float8E8M0FNU)
    sfb = cute.make_rmem_tensor((32,), cutlass.Float8E8M0FNU)
    for kb in range(K // 32):
        for j in cutlass.range_constexpr(cute.size(ra)):
            row = bm * 16 + ca[j][0]
            value = cutlass.Float32(0.0)
            if row < M:
                value = A[row * K + kb * 32 + ca[j][1]].to(cutlass.Float32)
            ra[j] = value.to(A.element_type)
        for j in cutlass.range_constexpr(cute.size(rb)):
            row = bn * 8 + cb[j][0]
            value = cutlass.Float32(0.0)
            if row < N:
                value = B[row * K + kb * 32 + cb[j][1]].to(cutlass.Float32)
            rb[j] = value.to(B.element_type)
        # SF atom TV layouts for m16n8k32: A ((2,2,8),(32,1))
        # strides ((8,0,1),(16,0)); B ((4,8),(32,1)), ((0,1),(8,0)).
        sr_a = bm * 16 + (lane % 2) * 8 + lane // 4
        sr_b = bn * 8 + lane // 4
        for j in cutlass.range_constexpr(32):
            if sr_a < M:
                sfa[j] = SA[sr_a * (K // 32) + kb]
            else:
                sfa[j] = cutlass.Float32(1.0).to(cutlass.Float8E8M0FNU)
            if sr_b < N:
                sfb[j] = SB[sr_b * (K // 32) + kb]
            else:
                sfb[j] = cutlass.Float32(1.0).to(cutlass.Float8E8M0FNU)
        mma.set(warp.Field.SFA, sfa.iterator)
        mma.set(warp.Field.SFB, sfb.iterator)
        cute.gemm(mma, acc, ra, rb, acc)
    for j in cutlass.range_constexpr(cute.size(acc)):
        row = bm * 16 + cc[j][0]
        col = bn * 8 + cc[j][1]
        if row < M and col < N:
            O[row * N + col] = acc[j]


@cute.jit
def _run(A: cute.Tensor, B: cute.Tensor, SA: cute.Tensor, SB: cute.Tensor,
         O: cute.Tensor, M: cutlass.Constexpr, N: cutlass.Constexpr,
         K: cutlass.Constexpr, stream: cuda.CUstream):
    op = warp.MmaMXF8Op(cutlass.Float8E4M3FN, cutlass.Float32, cutlass.Float8E8M0FNU)
    mma = cute.make_tiled_mma(op)
    _kernel(A, B, SA, SB, O, mma, M, N, K).launch(
        grid=((M+15)//16, (N+7)//8, 1), block=(32, 1, 1), stream=stream)


@cute.jit
def _stage_panel(src, scales, panel, s_scale, tx, row_start, k_start,
                 ROWS: cutlass.Constexpr, K: cutlass.Constexpr):
    # One vector per thread, two row bands cover a 32 x 128 panel.
    col = tx % 8 * 16
    vector = cute.make_layout(16)
    for i in cutlass.range_constexpr(2):
        row = tx // 8 + i * 16
        dst = cute.make_tensor(panel.iterator + cute.assume(row * 144 + col, divby=16), vector)
        if row_start + row < ROWS and k_start + col < K:
            offset = cute.assume((row_start + row) * K + k_start + col, divby=16)
            source = cute.make_tensor(src.iterator + offset, vector)
            cute.autovec_copy(source, dst)
        else:
            dst.fill(0)
    row, kb = tx // 4, tx % 4
    value = cutlass.Float32(1).to(cutlass.Float8E8M0FNU)
    if row_start + row < ROWS and k_start // 32 + kb < K // 32:
        value = scales[(row_start + row) * (K // 32) + k_start // 32 + kb]
    s_scale[row, kb] = value


@cute.kernel
def _kernel_tiled(A: cute.Tensor, B: cute.Tensor, SA: cute.Tensor, SB: cute.Tensor,
                  O: cute.Tensor, mma: cute.TiledMma,
                  M: cutlass.Constexpr, N: cutlass.Constexpr, K: cutlass.Constexpr):
    tx, _, _ = cute.arch.thread_idx()
    bm, bn, _ = cute.arch.block_idx()
    lane = tx % 32
    wm, wn = tx // 64 * 16, tx // 32 % 2 * 16
    thread = mma.get_slice(lane)
    ca = thread.partition_A(cute.make_identity_tensor((16, 32)))
    cb = thread.partition_B(cute.make_identity_tensor((8, 32)))
    cc = thread.partition_C(cute.make_identity_tensor((16, 8)))
    ra = cute.make_rmem_tensor_like(ca, A.element_type)
    rb = cute.make_rmem_tensor_like(cb, B.element_type)
    acc0 = cute.make_rmem_tensor_like(cc, cutlass.Float32)
    acc1 = cute.make_rmem_tensor_like(cc, cutlass.Float32)
    acc0.fill(0)
    acc1.fill(0)
    sfa = cute.make_rmem_tensor((32,), cutlass.Float8E8M0FNU)
    sfb = cute.make_rmem_tensor((32,), cutlass.Float8E8M0FNU)
    allocator = cutlass.memory.SmemAllocator()
    panel = cute.make_layout((32, 128), stride=(144, 1))
    scale_layout = cute.make_layout((32, 4), stride=(4, 1))
    a = allocator.allocate_tensor(A.element_type, panel, byte_alignment=16)
    b = allocator.allocate_tensor(B.element_type, panel, byte_alignment=16)
    a_scales = allocator.allocate_tensor(cutlass.Float8E8M0FNU, scale_layout, byte_alignment=16)
    b_scales = allocator.allocate_tensor(cutlass.Float8E8M0FNU, scale_layout, byte_alignment=16)
    for stage in range((K + 127) // 128):
        _stage_panel(A, SA, a, a_scales, tx, bm * 32, stage * 128, M, K)
        _stage_panel(B, SB, b, b_scales, tx, bn * 32, stage * 128, N, K)
        cute.arch.sync_threads()
        for kb in cutlass.range_constexpr(4):
            for j in cutlass.range_constexpr(cute.size(ra)):
                ra[j] = a[wm + ca[j][0], kb * 32 + ca[j][1]]
            sfa.fill(a_scales[wm + (lane % 2) * 8 + lane // 4, kb])
            mma.set(warp.Field.SFA, sfa.iterator)
            for j in cutlass.range_constexpr(cute.size(rb)):
                rb[j] = b[wn + cb[j][0], kb * 32 + cb[j][1]]
            sfb.fill(b_scales[wn + lane // 4, kb])
            mma.set(warp.Field.SFB, sfb.iterator)
            cute.gemm(mma, acc0, ra, rb, acc0)
            for j in cutlass.range_constexpr(cute.size(rb)):
                rb[j] = b[wn + 8 + cb[j][0], kb * 32 + cb[j][1]]
            sfb.fill(b_scales[wn + 8 + lane // 4, kb])
            mma.set(warp.Field.SFB, sfb.iterator)
            cute.gemm(mma, acc1, ra, rb, acc1)
        cute.arch.sync_threads()
    for j in cutlass.range_constexpr(cute.size(acc0)):
        row = bm * 32 + wm + cc[j][0]
        col = bn * 32 + wn + cc[j][1]
        if row < M:
            if col < N:
                O[row * N + col] = acc0[j]
            if col + 8 < N:
                O[row * N + col + 8] = acc1[j]


@cute.jit
def _run_tiled(A: cute.Tensor, B: cute.Tensor, SA: cute.Tensor, SB: cute.Tensor,
               O: cute.Tensor, M: cutlass.Constexpr, N: cutlass.Constexpr,
               K: cutlass.Constexpr, stream: cuda.CUstream):
    mma = cute.make_tiled_mma(warp.MmaMXF8Op(cutlass.Float8E4M3FN, cutlass.Float32, cutlass.Float8E8M0FNU))
    _kernel_tiled(A, B, SA, SB, O, mma, M, N, K).launch(
        grid=((M + 31) // 32, (N + 31) // 32, 1), block=(128, 1, 1), stream=stream)


_CACHE = {}
_QUANTIZER_CACHE = {}


@cute.jit
def _pack_block(bits):
    # Absolute BF16 bit patterns sort in the same order as finite magnitudes.
    maximum = bits & 0x7fff
    for shift in cutlass.range_constexpr(5):
        other = cute.arch.shuffle_sync_bfly(maximum, 1 << shift)
        maximum = other if other > maximum else maximum
    exponent = cutlass.Int32(0)
    if maximum != 0:
        exponent = (maximum >> 7).to(cutlass.Int32) - 135
        if (maximum & 0x7f) > 0x60:
            exponent += 1
        if exponent < -127:
            exponent = -127
    inverse = ((127 - exponent).to(cutlass.Uint32) << 23).bitcast(cutlass.Float32)
    value = (bits << 16).bitcast(cutlass.Float32) * inverse
    return value.to(cutlass.Float8E4M3FN), (exponent + 127).to(cutlass.Uint8)


@cute.kernel
def _quantize_bf16(X: cute.Tensor, O: cute.Tensor, S: cute.Tensor,
                   M: cutlass.Constexpr, K: cutlass.Constexpr, BLOCKS: cutlass.Constexpr,
                   SWIGLU: cutlass.Constexpr):
    tx, _, _ = cute.arch.thread_idx()
    bx, _, _ = cute.arch.block_idx()
    lane = tx % 32
    index = bx * 4 + tx // 32
    if index < M * BLOCKS:
        row, block = index // BLOCKS, index % BLOCKS
        col = block * 32 + lane
        bits = cutlass.Uint32(0)
        if col < K:
            if cutlass.const_expr(SWIGLU):
                base = row * (2 * K) + col
                value = (X[base].to(cutlass.Uint32) << 16).bitcast(cutlass.Float32)
                gate = (X[base + K].to(cutlass.Uint32) << 16).bitcast(cutlass.Float32)
                silu = (gate / (1.0 + cute.math.exp(-gate))).to(cutlass.BFloat16)
                product = (value * silu.to(cutlass.Float32)).to(cutlass.BFloat16)
                bits = product.bitcast(cutlass.Uint16).to(cutlass.Uint32)
            else:
                bits = X[row * K + col].to(cutlass.Uint32)
        code, scale = _pack_block(bits)
        O[index * 32 + lane] = code
        if lane == 0:
            S[index] = scale


@cute.jit
def _run_quantizer(X: cute.Tensor, O: cute.Tensor, S: cute.Tensor,
                   M: cutlass.Constexpr, K: cutlass.Constexpr,
                   SWIGLU: cutlass.Constexpr, stream: cuda.CUstream):
    blocks = (K + 31) // 32
    _quantize_bf16(X, O, S, M, K, blocks, SWIGLU).launch(
        grid=((M * blocks + 3) // 4, 1, 1), block=(128, 1, 1), stream=stream)


def launch_quantizer(x, *, swiglu=False):
    m, k = x.shape
    if swiglu:
        k //= 2
    blocks = (k + 31) // 32
    codes = torch.empty((m, blocks * 32), device=x.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty((m, blocks), device=x.device, dtype=torch.uint8)
    raw = x.detach().contiguous().view(torch.uint16)
    views = [from_dlpack(t.view(-1), assumed_align=t.element_size()) for t in (raw, codes, scales)]
    stream = cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
    key = (x.device.index, m, k, swiglu)
    compiled = _QUANTIZER_CACHE.get(key)
    if compiled is None:
        compiled = _QUANTIZER_CACHE[key] = cute.compile(_run_quantizer, *views, m, k, swiglu, stream)
    compiled(*views, stream)
    return codes, scales


def launch(a, b, *, tiled=False, optimized=None, bias=None, out=None, packed_b=False, producer_warps=0,
           tma=False, tma_store=False):
    if torch.cuda.get_device_capability(a.values.device) != (12, 0):
        raise RuntimeError('this experimental backend targets SM120 only')
    m, k = a.values.shape
    n = b.values.shape[0]
    if out is None:
        out = torch.empty((m, n), device=a.values.device, dtype=torch.float32)
    # Reinterpret raw bytes as unsigned E8M0; never numerically cast the bytes.
    sa, sb = (x.scales.view(torch.float8_e8m0fnu) for x in (a, b))
    tensors = (a.values, b.values, sa, sb, out)
    aligned_loads = tiled or optimized is not None
    if aligned_loads and (a.values.data_ptr() % 16 or b.values.data_ptr() % 16):
        raise ValueError('Tiled MXFP8 requires 16-byte-aligned payloads')
    if packed_b and (optimized is None or n % 32 or k % 128 or b.scales.data_ptr() % 16 or
                     b.scales.numel() != n * (k // 32)):
        raise ValueError('Packed MXFP8 requires optimized GEMM, N32/K128, and aligned scales')
    if producer_warps not in (0, 1, 2, 4) or (producer_warps and optimized is None):
        raise ValueError('Dedicated producer warps require optimized GEMM')
    if tma and producer_warps != 1:
        raise ValueError('TMA GEMM requires one dedicated producer warp')
    if tma_store and (not tma or optimized is None or out.dtype != torch.bfloat16 or
                      m % optimized[0] or n % optimized[1] or out.data_ptr() % 16 or not out.is_contiguous()):
        raise ValueError('TMA output prototype requires a TMA load schedule and full aligned BF16 tiles')
    aligned = ((0, 1, 3) if packed_b else (0, 1)) + ((4,) if tma_store else ())
    views = [from_dlpack(t.detach().view(-1), assumed_align=16 if aligned_loads and i in aligned else t.element_size())
             for i, t in enumerate(tensors)]
    if optimized is not None:
        views.append(None if bias is None else from_dlpack(bias.detach(), assumed_align=bias.element_size()))
    stream = cuda.CUstream(torch.cuda.current_stream(a.values.device).cuda_stream)
    key = (a.values.device.index, m, n, k, tiled, optimized, packed_b, producer_warps, tma, tma_store,
           out.dtype, None if bias is None else bias.dtype)
    compiled = _CACHE.get(key)
    if compiled is None:
        if optimized is not None:
            from ._cute_mxfp8_optimized import run
            config = (*optimized, 2, 2) if len(optimized) == 3 else optimized
            compiled = cute.compile(run, *views, m, n, k, *config, packed_b, producer_warps, tma, tma_store, stream)
        else:
            compiled = cute.compile(_run_tiled if tiled else _run, *views, m, n, k, stream)
        _CACHE[key] = compiled
    compiled(*views, stream)
    return out
