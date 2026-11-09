"""SM120 FFN projection pipeline with independent operand and epilogue rings."""
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings import driver as cuda
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.cute.runtime import from_dlpack

from ._cute_mxfp8_optimized import _stage_packed_scales, _stage_scales
from ._cute_warp_pipeline import acquire_slot, arrive_warp, init_ring, wait_slot
from .mxfp8 import MXFP8Tensor


@dataclass(frozen=True)
class Schedule:
    """CTA shape, K ring depth, and down-projection writeback ring depth."""
    rows: int = 128
    cols: int = 128
    k_slots: int = 2
    output_slots: int = 1
    ctas_per_sm: int = 0


@cute.jit
def _swiglu_bits(value, gate):
    silu = (gate / (1.0 + cute.math.exp(-gate))).to(cutlass.BFloat16)
    product = (value * silu.to(cutlass.Float32)).to(cutlass.BFloat16)
    return product.bitcast(cutlass.Uint16).to(cutlass.Uint32)


@cute.jit
def _block_scale(maximum):
    exponent = cutlass.Int32(0)
    if maximum != 0:
        # Match the BF16 quantizer's exact 0.875 frexp threshold.
        exponent = (maximum >> 7).to(cutlass.Int32) - 135
        if (maximum & 0x7f) > 0x60:
            exponent += 1
        if exponent < -127:
            exponent = -127
    inverse = ((127 - exponent).to(cutlass.Uint32) << 23).bitcast(cutlass.Float32)
    return inverse, (exponent + 127).to(cutlass.Uint8)


@cute.jit
def _fp8(bits, inverse):
    return ((bits << 16).bitcast(cutlass.Float32) * inverse).to(cutlass.Float8E4M3FN)


@cute.jit
def _register_epilogue(accum, cc, bias, Y, YS, lane, first_row, first_col,
                       M: cutlass.Constexpr, N: cutlass.Constexpr, MATOMS: cutlass.Constexpr):
    # The m16n8 accumulator layout assigns each row to four adjacent lanes.
    # Each warp's N64 fragment pairs one N32 value group with its N32 gate.
    bits = cute.make_rmem_tensor((2, 4), cutlass.Uint32)
    for i in cutlass.range_constexpr(MATOMS):
        for row_fragment in cutlass.range_constexpr(2):
            maximum = cutlass.Uint32(0)
            for j in cutlass.range_constexpr(4):
                for c in cutlass.range_constexpr(2):
                    p = row_fragment * 2 + c
                    value = accum[p, i, j]
                    gate = accum[p, i, j + 4]
                    if cutlass.const_expr(bias is not None):
                        col = first_col + j * 8 + cc[p][1]
                        value += bias[col].to(cutlass.Float32)
                        gate += bias[col + N // 2].to(cutlass.Float32)
                    entry = _swiglu_bits(value.to(cutlass.BFloat16).to(cutlass.Float32),
                                         gate.to(cutlass.BFloat16).to(cutlass.Float32))
                    bits[c, j] = entry
                    magnitude = entry & 0x7fff
                    maximum = magnitude if magnitude > maximum else maximum
            for shift in cutlass.range_constexpr(2):
                other = cute.arch.shuffle_sync_bfly(maximum, 1 << shift)
                maximum = other if other > maximum else maximum
            inverse, scale = _block_scale(maximum)
            row = first_row + i * 16 + cc[row_fragment * 2][0]
            if row < M:
                dst = row * (N // 2) + first_col
                for j in cutlass.range_constexpr(4):
                    for c in cutlass.range_constexpr(2):
                        p = row_fragment * 2 + c
                        Y[dst + j * 8 + cc[p][1]] = _fp8(bits[c, j], inverse)
                if lane % 4 == 0:
                    YS[dst // 32] = scale


@cute.kernel
def _projection(SA, SB, Y, YS, bias, tma_a, ga, tma_b, gb, tma_out, go, mma,
                M: cutlass.Constexpr, N: cutlass.Constexpr, K: cutlass.Constexpr,
                TM: cutlass.Constexpr, TN: cutlass.Constexpr, KS: cutlass.Constexpr,
                ES: cutlass.Constexpr, GRID: cutlass.Constexpr):
    tx, _, _ = cute.arch.thread_idx()
    cta, _, _ = cute.arch.block_idx()
    lane, warp_id = tx % 32, cute.arch.warp_idx()
    m_tiles = (M + TM - 1) // TM
    n_tiles = N // TN
    k_tiles = K // 128
    smem = cutlass.memory.SmemAllocator()
    swizzle = cute.make_swizzle(3, 4, 3)
    a = smem.allocate_tensor(cutlass.Float8E4M3FN,
                             cute.make_layout((TM, 128, KS), stride=(128, 1, TM * 128)),
                             byte_alignment=1024, swizzle=swizzle)
    b = smem.allocate_tensor(cutlass.Float8E4M3FN,
                             cute.make_layout((TN, 128, KS), stride=(128, 1, TN * 128)),
                             byte_alignment=1024, swizzle=swizzle)
    sa = smem.allocate_tensor(cutlass.Float8E8M0FNU, cute.make_layout((TM, 4, KS), stride=(4, 1, TM * 4)))
    sb = smem.allocate_tensor(cutlass.Float8E8M0FNU, cute.make_layout((TN, 4, KS), stride=(4, 1, TN * 4)),
                              byte_alignment=16)
    if cutlass.const_expr(tma_out is not None):
        out = smem.allocate_tensor(cutlass.BFloat16,
                                   cute.make_layout((TM, TN, ES), stride=(TN, 1, TM * TN)),
                                   byte_alignment=128)
    k_ready, k_free = init_ring(smem, tx, KS, 1, 4)
    if cutlass.const_expr(tma_out is not None):
        out_ready, out_free = init_ring(smem, tx, ES, 4, 1)

    if warp_id == 0:
        # The loader can prefetch the next tile before the current epilogue retires.
        ticket = cutlass.Int32(0)
        for tile in range(cta, m_tiles * n_tiles, GRID):
            bm, bn = tile % m_tiles, tile // m_tiles
            a_tile = cute.local_tile(ga, (TM, 128), (bm, None))
            if cutlass.const_expr(tma_out is None):
                half = TN // 2
                b_layout = cute.make_layout((half, 128, KS), stride=(128, 1, TN * 128))
                bv = cute.make_tensor(b.iterator, b_layout)
                bg = cute.make_tensor(b.iterator + half * 128, b_layout)
                values = cute.local_tile(gb, (half, 128), (bn, None))
                gates = cute.local_tile(gb, (half, 128), (bn + n_tiles, None))
            else:
                values = cute.local_tile(gb, (TN, 128), (bn, None))
                bv = b
            one = cute.make_layout(1)
            a_shared, a_global = cpasync.tma_partition(tma_a, 0, one, cute.group_modes(a, 0, 2),
                                                       cute.group_modes(a_tile, 0, 2))
            b_shared, b_global = cpasync.tma_partition(tma_b, 0, one, cute.group_modes(bv, 0, 2),
                                                       cute.group_modes(values, 0, 2))
            if cutlass.const_expr(tma_out is None):
                g_shared, g_global = cpasync.tma_partition(tma_b, 0, one, cute.group_modes(bg, 0, 2),
                                                           cute.group_modes(gates, 0, 2))
            for k_step in range(k_tiles):
                acquire_slot(k_free, ticket, KS)
                slot = ticket % KS
                first_k = k_step * 128
                _stage_scales(SA, sa[None, None, slot], lane, bm * TM, first_k, M, K, TM, 32)
                if cutlass.const_expr(tma_out is None):
                    # One lane stages four rows. The upper lanes load the gate half.
                    offset = cute.assume(slot * TN * 4 + (lane // (half // 4)) * (half * 4), divby=16)
                    bsc = cute.make_tensor(sb.iterator + offset,
                                           cute.make_layout((half, 4), stride=(4, 1)))
                    if lane < TN // 4:
                        first_n = (bn + lane // (half // 4) * n_tiles) * half
                        _stage_packed_scales(SB, bsc, lane % (half // 4), first_n, first_k, N, K, half)
                else:
                    _stage_packed_scales(SB, sb[None, None, slot], lane, bn * TN, first_k, N, K, TN)
                cute.arch.sync_warp()
                barrier = k_ready.iterator + slot
                if lane == 0:
                    cute.arch.mbarrier_arrive_and_expect_tx(barrier, (TM + TN) * 128)
                cute.copy(tma_a, a_global[None, k_step], a_shared[None, slot], tma_bar_ptr=barrier)
                cute.copy(tma_b, b_global[None, k_step], b_shared[None, slot], tma_bar_ptr=barrier)
                if cutlass.const_expr(tma_out is None):
                    cute.copy(tma_b, g_global[None, k_step], g_shared[None, slot], tma_bar_ptr=barrier)
                ticket += 1
    elif warp_id <= 4:
        compute_warp = warp_id - 1
        wm, wn = compute_warp // 2 * (TM // 2), compute_warp % 2 * (TN // 2)
        m_atoms, n_atoms = TM // 32, TN // 16
        b_rows = cute.make_rmem_tensor(n_atoms, cutlass.Int32)
        for j in cutlass.range_constexpr(n_atoms):
            if cutlass.const_expr(tma_out is None):
                b_rows[j] = wn // 2 + (j % 4) * 8 + (j // 4) * (TN // 2)
            else:
                b_rows[j] = wn + j * 8
        thread = mma.get_slice(lane)
        ca = thread.partition_A(cute.make_identity_tensor((16, 32)))
        cb = thread.partition_B(cute.make_identity_tensor((8, 32)))
        cc = thread.partition_C(cute.make_identity_tensor((16, 8)))
        ra = cute.make_rmem_tensor_like(ca, a.element_type)
        rb = cute.make_rmem_tensor((cb.shape, n_atoms), b.element_type)
        accum = cute.make_rmem_tensor((cc.shape, m_atoms, n_atoms), cutlass.Float32)
        sfa = cute.make_rmem_tensor(32, cutlass.Float8E8M0FNU)
        sfb = cute.make_rmem_tensor(32, cutlass.Float8E8M0FNU)
        copy_a = cute.make_tiled_copy_A(cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, 4), a.element_type), mma)
        copy_b = cute.make_tiled_copy_B(cute.make_copy_atom(warp.LdMatrix8x8x16bOp(False, 2), b.element_type), mma)
        thread_a, thread_b = copy_a.get_slice(lane), copy_b.get_slice(lane)
        copy_ra = thread_a.retile(ra)
        ticket = cutlass.Int32(0)
        tile_ticket = cutlass.Int32(0)
        for tile in range(cta, m_tiles * n_tiles, GRID):
            accum.fill(0)
            for k_step in range(k_tiles):
                wait_slot(k_ready, ticket, KS)
                slot = ticket % KS
                for kb in cutlass.range_constexpr(4):
                    for j in cutlass.range_constexpr(n_atoms):
                        src = cute.local_tile(b[None, None, slot], (8, 32), (b_rows[j] // 8, kb))
                        dst = cute.append_ones(rb[None, j], up_to_rank=3)
                        cute.copy(copy_b, thread_b.partition_S(src), thread_b.retile(dst))
                    for i in cutlass.range_constexpr(m_atoms):
                        row = wm + i * 16
                        src = cute.local_tile(a[None, None, slot], (16, 32), (wm // 16 + i, kb))
                        cute.copy(copy_a, thread_a.partition_S(src), copy_ra)
                        sfa.fill(sa[row + (lane % 2) * 8 + lane // 4, kb, slot])
                        mma.set(warp.Field.SFA, sfa.iterator)
                        for j in cutlass.range_constexpr(n_atoms):
                            sfb.fill(sb[b_rows[j] + lane // 4, kb, slot])
                            mma.set(warp.Field.SFB, sfb.iterator)
                            c = cute.append_ones(accum[None, i, j], up_to_rank=3)
                            cute.gemm(mma, c, ra, cute.append_ones(rb[None, j], up_to_rank=3), c)
                cute.arch.fence_proxy("async.shared", space="cta")
                arrive_warp(k_free, lane, ticket, KS)
                ticket += 1
            if cutlass.const_expr(tma_out is None):
                first_row = (tile % m_tiles) * TM + wm
                first_col = (tile // m_tiles) * (TN // 2) + wn // 2
                _register_epilogue(accum, cc, bias, Y, YS, lane, first_row, first_col, M, N, m_atoms)
            else:
                acquire_slot(out_free, tile_ticket, ES)
                for i in cutlass.range_constexpr(m_atoms):
                    for j in cutlass.range_constexpr(n_atoms):
                        for p in cutlass.range_constexpr(cute.size(cc)):
                            row = wm + i * 16 + cc[p][0]
                            col = wn + j * 8 + cc[p][1]
                            value = accum[p, i, j]
                            if cutlass.const_expr(bias is not None):
                                value += bias[tile // m_tiles * TN + col].to(cutlass.Float32)
                            out[row, col, tile_ticket % ES] = value.to(cutlass.BFloat16)
                cute.arch.fence_proxy("async.shared", space="cta")
                arrive_warp(out_ready, lane, tile_ticket, ES)
                tile_ticket += 1
    elif cutlass.const_expr(tma_out is not None):
        tile_ticket = cutlass.Int32(0)
        for tile in range(cta, m_tiles * n_tiles, GRID):
            bm, bn = tile % m_tiles, tile // m_tiles
            wait_slot(out_ready, tile_ticket, ES)
            g_tile = cute.local_tile(go, (TM, TN), (bm, bn))
            shared, dest = cpasync.tma_partition(tma_out, 0, cute.make_layout(1),
                                                cute.group_modes(out[None, None, tile_ticket % ES], 0, 2),
                                                cute.group_modes(g_tile, 0, 2))
            cute.copy(tma_out, shared, dest)
            cute.arch.cp_async_bulk_commit_group()
            # Release the shared slot when its source has been read. Drain global
            # writes before exit so the next kernel can consume the BF16 result.
            cute.arch.cp_async_bulk_wait_group(0, read=True)
            arrive_warp(out_free, lane, tile_ticket, ES)
            tile_ticket += 1
        cute.arch.cp_async_bulk_wait_group(0)


@cute.jit
def _run(A, B, SA, SB, Y, YS, bias,
         M: cutlass.Constexpr, N: cutlass.Constexpr, K: cutlass.Constexpr,
         TM: cutlass.Constexpr, TN: cutlass.Constexpr, KS: cutlass.Constexpr,
         ES: cutlass.Constexpr, GRID: cutlass.Constexpr, stream: cuda.CUstream):
    swizzle = cute.make_swizzle(3, 4, 3)
    a = cute.make_tensor(A.iterator, cute.make_layout((M, K), stride=(K, 1)))
    b = cute.make_tensor(B.iterator, cute.make_layout((N, K), stride=(K, 1)))
    a_layout = cute.make_composed_layout(swizzle, 0, cute.make_layout((TM, 128), stride=(128, 1)))
    b_rows = TN // 2 if YS is not None else TN
    b_layout = cute.make_composed_layout(swizzle, 0, cute.make_layout((b_rows, 128), stride=(128, 1)))
    tma_a, ga = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), a, a_layout, (TM, 128))
    tma_b, gb = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), b, b_layout, (b_rows, 128))
    tma_out, go = None, None
    if cutlass.const_expr(YS is None):
        out = cute.make_tensor(Y.iterator, cute.make_layout((M, N), stride=(N, 1)))
        out_layout = cute.make_layout((TM, TN), stride=(TN, 1))
        tma_out, go = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileS2GOp(), out, out_layout, (TM, TN))
    mma = cute.make_tiled_mma(warp.MmaMXF8Op(cutlass.Float8E4M3FN, cutlass.Float32, cutlass.Float8E8M0FNU))
    _projection(SA, SB, Y, YS, bias, tma_a, ga, tma_b, gb, tma_out, go, mma,
                M, N, K, TM, TN, KS, ES, GRID).launch(
        grid=(GRID, 1, 1), block=(32 * (6 if YS is None else 5), 1, 1), stream=stream)


_CACHE = {}


def launch_up(a, weight, bias, schedule=Schedule(), *, output=None):
    m = a.values.shape[0]
    n = weight.values.shape[0]
    device = a.values.device
    if schedule.cols != 128:
        raise ValueError("Register epilogue requires a 128-column tile")
    if output is None:
        values = torch.empty((m, n // 2), dtype=torch.float8_e4m3fn, device=device)
        scales = torch.empty((m, n // 64), dtype=torch.uint8, device=device)
        output = MXFP8Tensor(values, scales, n // 2)
    _launch(a, weight, output.values, output.scales, bias, schedule)
    return output


def launch_down(a, weight, bias, output, schedule):
    _launch(a, weight, output, None, bias, schedule)
    return output


def _launch(a, weight, output, scales, bias, schedule):
    m, k = a.values.shape
    n = weight.values.shape[0]
    if n % schedule.cols or k % 128:
        raise ValueError("Fused MXFP8 FFN requires aligned output and K dimensions")
    grid = ((m + schedule.rows - 1) // schedule.rows) * (n // schedule.cols)
    device = a.values.device
    if schedule.ctas_per_sm:
        grid = min(grid, schedule.ctas_per_sm * torch.cuda.get_device_properties(device).multi_processor_count)
    tensors = (a.values, weight.values, a.scales.view(torch.float8_e8m0fnu),
               weight.scales.view(torch.float8_e8m0fnu), output, scales, bias)
    aligned = (0, 1, 3, 4) if scales is None else (0, 1, 3)
    views = [None if t is None else from_dlpack(t.detach().view(-1), assumed_align=16 if i in aligned else t.element_size())
             for i, t in enumerate(tensors)]
    stream = cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)
    key = (device.index, m, n, k, schedule, grid, scales is None, None if bias is None else bias.dtype)
    compiled = _CACHE.get(key)
    if compiled is None:
        compiled = cute.compile(_run, *views, m, n, k, schedule.rows, schedule.cols, schedule.k_slots,
                                schedule.output_slots, grid, stream)
        _CACHE[key] = compiled
    compiled(*views, stream)
