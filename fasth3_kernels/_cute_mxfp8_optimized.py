"""SM120 MXFP8 GEMM with warp matrix loads and optional async K staging."""
import cutlass
import cutlass.cute as cute
from cuda.bindings import driver as cuda
from cutlass.cute.nvgpu import cpasync, warp

from ._cute_warp_pipeline import acquire_slot, arrive_warp, init_ring, wait_slot


_K_TILE = 128
_PITCH = 144


@cute.jit
def _stage_scales(scales, panel, tx, first_row, first_k,
                  ROWS: cutlass.Constexpr, K: cutlass.Constexpr,
                  TILE: cutlass.Constexpr, THREADS: cutlass.Constexpr):
    for i in cutlass.range_constexpr((TILE + THREADS // 4 - 1) // (THREADS // 4)):
        row, block = tx // 4 + i * (THREADS // 4), tx % 4
        k_block = first_k // 32 + block
        if row < TILE:
            scale = cutlass.Float32(1).to(cutlass.Float8E8M0FNU)
            if first_row + row < ROWS and k_block < K // 32:
                scale = scales[(first_row + row) * (K // 32) + k_block]
            panel[row, block] = scale


@cute.jit
def _stage(src, scales, panel, scale_panel, tx, first_row, first_k,
           ROWS: cutlass.Constexpr, K: cutlass.Constexpr, TILE: cutlass.Constexpr,
           THREADS: cutlass.Constexpr, async_copy=None):
    col = tx % 8 * 16
    vector = cute.make_layout(16)
    for i in cutlass.range_constexpr(TILE // (THREADS // 8)):
        row = tx // 8 + i * (THREADS // 8)
        offset = cute.assume(row * _PITCH + col, divby=16)
        dst = cute.make_tensor(panel.iterator + offset, vector)
        if first_row + row < ROWS and first_k + col < K:
            offset = cute.assume((first_row + row) * K + first_k + col, divby=16)
            source = cute.make_tensor(src.iterator + offset, vector)
            if cutlass.const_expr(async_copy is None):
                cute.autovec_copy(source, dst)
            else:
                cute.copy(async_copy, source, dst)
        else:
            dst.fill(0)
    if cutlass.const_expr(scale_panel is not None):
        _stage_scales(scales, scale_panel, tx, first_row, first_k, ROWS, K, TILE, THREADS)


@cute.jit
def _stage_packed_scales(SB, panel, tx, first_row, first_k,
                         N: cutlass.Constexpr, K: cutlass.Constexpr, TN: cutlass.Constexpr, async_copy=None):
    # One 16-byte copy supplies four N rows and four K32 scales per row.
    if tx < TN // 4:
        row = first_row + tx * 4
        layout = cute.make_layout(16)
        offset = cute.assume(tx * 16, divby=16)
        dest = cute.make_tensor(panel.iterator + offset, layout)
        if row < N:
            packed = (row // 32 * (K // 128) + first_k // 128) * 128 + row % 32 * 4
            offset = cute.assume(packed, divby=16)
            source = cute.make_tensor(SB.iterator + offset, layout)
            if cutlass.const_expr(async_copy is None):
                cute.autovec_copy(source, dest)
            else:
                scale_copy = cute.make_copy_atom(cpasync.CopyG2SOp(), SB.element_type, num_bits_per_copy=128)
                cute.copy(scale_copy, source, dest)
        else:
            dest.fill(cutlass.Float32(1).to(cutlass.Float8E8M0FNU))


@cute.jit
def _stage_operands(A, B, SA, SB, a_panel, b_panel, a_scales, b_scales, tx, bm, bn,
                    first_k, slot, M: cutlass.Constexpr, N: cutlass.Constexpr, K: cutlass.Constexpr,
                    TM: cutlass.Constexpr, TN: cutlass.Constexpr, THREADS: cutlass.Constexpr,
                    PACKED_B: cutlass.Constexpr, async_copy=None):
    _stage(A, SA, a_panel[None, None, slot], a_scales[None, None, slot],
           tx, bm * TM, first_k, M, K, TM, THREADS, async_copy)
    scales = b_scales[None, None, slot]
    if cutlass.const_expr(PACKED_B):
        _stage(B, SB, b_panel[None, None, slot], None, tx, bn * TN, first_k, N, K, TN, THREADS, async_copy)
        _stage_packed_scales(SB, scales, tx, bn * TN, first_k, N, K, TN, async_copy)
    else:
        _stage(B, SB, b_panel[None, None, slot], scales, tx, bn * TN, first_k, N, K, TN, THREADS, async_copy)


@cute.kernel
def _kernel(A: cute.Tensor, B: cute.Tensor, SA: cute.Tensor, SB: cute.Tensor, O: cute.Tensor, bias,
            tma_a, tma_b, gmem_a, gmem_b, tma_o, gmem_o,
            mma: cute.TiledMma, M: cutlass.Constexpr, N: cutlass.Constexpr, K: cutlass.Constexpr,
            TM: cutlass.Constexpr, TN: cutlass.Constexpr, STAGES: cutlass.Constexpr,
            WARPS_M: cutlass.Constexpr, WARPS_N: cutlass.Constexpr, PACKED_B: cutlass.Constexpr, PRODUCERS: cutlass.Constexpr):
    tx, _, _ = cute.arch.thread_idx()
    bm, bn, _ = cute.arch.block_idx()
    lane, warp_id = tx % 32, tx // 32
    threads = 32 * WARPS_M * WARPS_N
    use_tma = cutlass.const_expr(tma_a is not None)
    allocator = cutlass.memory.SmemAllocator()
    if cutlass.const_expr(tma_o is not None):
        s_out = allocator.allocate_tensor(O.element_type, cute.make_layout((TM, TN), stride=(TN, 1)),
                                          byte_alignment=128)
    if cutlass.const_expr(use_tma):
        swizzle = cute.make_swizzle(3, 4, 3)
        a_layout = cute.make_layout((TM, _K_TILE, STAGES), stride=(_K_TILE, 1, TM * _K_TILE))
        b_layout = cute.make_layout((TN, _K_TILE, STAGES), stride=(_K_TILE, 1, TN * _K_TILE))
        a_panel = allocator.allocate_tensor(A.element_type, a_layout, byte_alignment=1024, swizzle=swizzle)
        b_panel = allocator.allocate_tensor(B.element_type, b_layout, byte_alignment=1024, swizzle=swizzle)
    else:
        a_layout = cute.make_layout((TM, _K_TILE, STAGES), stride=(_PITCH, 1, TM * _PITCH))
        b_layout = cute.make_layout((TN, _K_TILE, STAGES), stride=(_PITCH, 1, TN * _PITCH))
        a_panel = allocator.allocate_tensor(A.element_type, a_layout, byte_alignment=16)
        b_panel = allocator.allocate_tensor(B.element_type, b_layout, byte_alignment=16)
    a_scales = allocator.allocate_tensor(
        cutlass.Float8E8M0FNU, cute.make_layout((TM, 4, STAGES), stride=(4, 1, TM * 4)))
    b_scales = allocator.allocate_tensor(
        cutlass.Float8E8M0FNU, cute.make_layout((TN, 4, STAGES), stride=(4, 1, TN * 4)), byte_alignment=16)
    k_tiles = (K + _K_TILE - 1) // _K_TILE
    if cutlass.const_expr(PRODUCERS > 0):
        ready, free = init_ring(allocator, tx, STAGES, PRODUCERS, WARPS_M * WARPS_N)
    if tx < threads:
        wm, wn = warp_id // WARPS_N * (TM // WARPS_M), warp_id % WARPS_N * (TN // WARPS_N)
        m_atoms, n_atoms = TM // (16 * WARPS_M), TN // (8 * WARPS_N)
        thread = mma.get_slice(lane)
        ca = thread.partition_A(cute.make_identity_tensor((16, 32)))
        cb = thread.partition_B(cute.make_identity_tensor((8, 32)))
        cc = thread.partition_C(cute.make_identity_tensor((16, 8)))
        ra = cute.make_rmem_tensor_like(ca, A.element_type)
        rb = cute.make_rmem_tensor((cb.shape, n_atoms), B.element_type)
        accum = cute.make_rmem_tensor((cc.shape, m_atoms, n_atoms), cutlass.Float32)
        accum.fill(0)
        sfa = cute.make_rmem_tensor((32,), cutlass.Float8E8M0FNU)
        sfb = cute.make_rmem_tensor((32,), cutlass.Float8E8M0FNU)
        copy_a = cute.make_tiled_copy_A(
            cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, 4), A.element_type), mma)
        copy_b = cute.make_tiled_copy_B(
            cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, 2), B.element_type), mma)
        thread_a, thread_b = copy_a.get_slice(lane), copy_b.get_slice(lane)
        copy_ra = thread_a.retile(ra)
        a_view = cute.make_layout((16, 32), stride=(_PITCH, 1))
        b_view = cute.make_layout((8, 32), stride=(_PITCH, 1))

        if cutlass.const_expr(STAGES == 2 and PRODUCERS == 0):
            async_copy = cute.make_copy_atom(cpasync.CopyG2SOp(), A.element_type, num_bits_per_copy=128)
            _stage_operands(A, B, SA, SB, a_panel, b_panel, a_scales, b_scales,
                            tx, bm, bn, 0, 0, M, N, K, TM, TN, threads, PACKED_B, async_copy)
            cute.arch.cp_async_commit_group()
        for stage in range(k_tiles):
            slot = stage % STAGES
            if cutlass.const_expr(PRODUCERS > 0):
                wait_slot(ready, stage, STAGES)
            elif cutlass.const_expr(STAGES == 2):
                if stage + 1 < k_tiles:
                    _stage_operands(A, B, SA, SB, a_panel, b_panel, a_scales, b_scales,
                                    tx, bm, bn, (stage + 1) * _K_TILE, 1 - slot, M, N, K, TM, TN, threads,
                                    PACKED_B, async_copy)
                    cute.arch.cp_async_commit_group()
                    cute.arch.cp_async_wait_group(1)
                else:
                    cute.arch.cp_async_wait_group(0)
            else:
                _stage_operands(A, B, SA, SB, a_panel, b_panel, a_scales, b_scales,
                                tx, bm, bn, stage * _K_TILE, 0, M, N, K, TM, TN, threads, PACKED_B)
            if cutlass.const_expr(PRODUCERS == 0):
                cute.arch.sync_threads()
            a = a_panel[None, None, slot]
            b = b_panel[None, None, slot]
            for kb in cutlass.range_constexpr(4):
                # Reuse each B fragment for every M atom owned by the warp.
                for j in cutlass.range_constexpr(n_atoms):
                    if cutlass.const_expr(use_tma):
                        src = cute.local_tile(b, (8, 32), (wn // 8 + j, kb))
                    else:
                        offset = cute.assume((wn + j * 8) * _PITCH + kb * 32, divby=16)
                        src = cute.make_tensor(b.iterator + offset, b_view)
                    dst = cute.append_ones(rb[None, j], up_to_rank=3)
                    cute.copy(copy_b, thread_b.partition_S(src), thread_b.retile(dst))
                for i in cutlass.range_constexpr(m_atoms):
                    row = wm + i * 16
                    if cutlass.const_expr(use_tma):
                        src = cute.local_tile(a, (16, 32), (wm // 16 + i, kb))
                    else:
                        offset = cute.assume(row * _PITCH + kb * 32, divby=16)
                        src = cute.make_tensor(a.iterator + offset, a_view)
                    cute.copy(copy_a, thread_a.partition_S(src), copy_ra)
                    sfa.fill(a_scales[row + (lane % 2) * 8 + lane // 4, kb, slot])
                    mma.set(warp.Field.SFA, sfa.iterator)
                    for j in cutlass.range_constexpr(n_atoms):
                        sfb.fill(b_scales[wn + j * 8 + lane // 4, kb, slot])
                        mma.set(warp.Field.SFB, sfb.iterator)
                        c = cute.append_ones(accum[None, i, j], up_to_rank=3)
                        cute.gemm(mma, c, ra, cute.append_ones(rb[None, j], up_to_rank=3), c)
            if cutlass.const_expr(PRODUCERS > 0):
                if cutlass.const_expr(use_tma):
                    cute.arch.fence_proxy("async.shared", space="cta")
                arrive_warp(free, lane, stage, STAGES)
            else:
                cute.arch.sync_threads()

        for i in cutlass.range_constexpr(m_atoms):
            for j in cutlass.range_constexpr(n_atoms):
                for p in cutlass.range_constexpr(cute.size(cc)):
                    row = bm * TM + wm + i * 16 + cc[p][0]
                    col = bn * TN + wn + j * 8 + cc[p][1]
                    if row < M and col < N:
                        value = accum[p, i, j]
                        if cutlass.const_expr(bias is not None):
                            value += bias[col].to(cutlass.Float32)
                        if cutlass.const_expr(tma_o is not None):
                            s_out[row - bm * TM, col - bn * TN] = value.to(O.element_type)
                        else:
                            O[row * N + col] = value.to(O.element_type)
    elif cutlass.const_expr(PRODUCERS > 0):
        async_copy = cute.make_copy_atom(cpasync.CopyG2SOp(), A.element_type, num_bits_per_copy=128)
        if cutlass.const_expr(use_tma):
            ga = cute.local_tile(gmem_a, (TM, _K_TILE), (bm, None))
            gb = cute.local_tile(gmem_b, (TN, _K_TILE), (bn, None))
            one = cute.make_layout(1)
            sa_tma, ga_tma = cpasync.tma_partition(tma_a, 0, one, cute.group_modes(a_panel, 0, 2),
                                                   cute.group_modes(ga, 0, 2))
            sb_tma, gb_tma = cpasync.tma_partition(tma_b, 0, one, cute.group_modes(b_panel, 0, 2),
                                                   cute.group_modes(gb, 0, 2))
        for stage in range(k_tiles):
            acquire_slot(free, stage, STAGES)
            slot = stage % STAGES
            if cutlass.const_expr(use_tma):
                first_k = stage * _K_TILE
                _stage_scales(SA, a_scales[None, None, slot], lane, bm * TM, first_k, M, K, TM, 32)
                if cutlass.const_expr(PACKED_B):
                    _stage_packed_scales(SB, b_scales[None, None, slot], lane, bn * TN, first_k, N, K, TN)
                else:
                    _stage_scales(SB, b_scales[None, None, slot], lane, bn * TN, first_k, N, K, TN, 32)
                cute.arch.sync_warp()
                barrier = ready.iterator + slot
                if lane == 0:
                    cute.arch.mbarrier_arrive_and_expect_tx(barrier, (TM + TN) * _K_TILE)
                # CuTe emits a warp-wide election for each TMA instruction.
                cute.copy(tma_a, ga_tma[None, stage], sa_tma[None, slot], tma_bar_ptr=barrier)
                cute.copy(tma_b, gb_tma[None, stage], sb_tma[None, slot], tma_bar_ptr=barrier)
            else:
                _stage_operands(A, B, SA, SB, a_panel, b_panel, a_scales, b_scales, tx - threads, bm, bn,
                                stage * _K_TILE, slot, M, N, K, TM, TN, PRODUCERS * 32, PACKED_B, async_copy)
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(0)
                arrive_warp(ready, lane, stage, STAGES)
    if cutlass.const_expr(tma_o is not None):
        # Every compute warp publishes its accumulator fragment to the async
        # proxy. The existing load warp issues the store after the K loop.
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.sync_threads()
        if warp_id == WARPS_M * WARPS_N:
            g_tile = cute.local_tile(gmem_o, (TM, TN), (bm, bn))
            shared, dest = cpasync.tma_partition(tma_o, 0, cute.make_layout(1),
                                                cute.group_modes(s_out, 0, 2),
                                                cute.group_modes(g_tile, 0, 2))
            cute.copy(tma_o, shared, dest)
            cute.arch.cp_async_bulk_commit_group()
            cute.arch.cp_async_bulk_wait_group(0)



@cute.jit
def run(A: cute.Tensor, B: cute.Tensor, SA: cute.Tensor, SB: cute.Tensor, O: cute.Tensor, bias,
        M: cutlass.Constexpr, N: cutlass.Constexpr, K: cutlass.Constexpr,
        TM: cutlass.Constexpr, TN: cutlass.Constexpr, STAGES: cutlass.Constexpr,
        WARPS_M: cutlass.Constexpr, WARPS_N: cutlass.Constexpr, PACKED_B: cutlass.Constexpr,
        PRODUCERS: cutlass.Constexpr, TMA: cutlass.Constexpr, TMA_STORE: cutlass.Constexpr,
        stream: cuda.CUstream):
    op = warp.MmaMXF8Op(cutlass.Float8E4M3FN, cutlass.Float32, cutlass.Float8E8M0FNU)
    mma = cute.make_tiled_mma(op)
    tma_a, tma_b, ga, gb = None, None, None, None
    tma_o, go = None, None
    if cutlass.const_expr(TMA):
        swizzle = cute.make_swizzle(3, 4, 3)
        a = cute.make_tensor(A.iterator, cute.make_layout((M, K), stride=(K, 1)))
        b = cute.make_tensor(B.iterator, cute.make_layout((N, K), stride=(K, 1)))
        a_layout = cute.make_composed_layout(swizzle, 0, cute.make_layout((TM, _K_TILE), stride=(_K_TILE, 1)))
        b_layout = cute.make_composed_layout(swizzle, 0, cute.make_layout((TN, _K_TILE), stride=(_K_TILE, 1)))
        tma_a, ga = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), a, a_layout, (TM, _K_TILE))
        tma_b, gb = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), b, b_layout, (TN, _K_TILE))
    if cutlass.const_expr(TMA_STORE):
        out = cute.make_tensor(O.iterator, cute.make_layout((M, N), stride=(N, 1)))
        tma_o, go = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileS2GOp(), out,
                                               cute.make_layout((TM, TN), stride=(TN, 1)), (TM, TN))
    _kernel(A, B, SA, SB, O, bias, tma_a, tma_b, ga, gb, tma_o, go, mma, M, N, K, TM, TN, STAGES,
            WARPS_M, WARPS_N, PACKED_B, PRODUCERS).launch(
        grid=((M + TM - 1) // TM, (N + TN - 1) // TN, 1),
        block=(32 * (WARPS_M * WARPS_N + PRODUCERS), 1, 1), stream=stream)
