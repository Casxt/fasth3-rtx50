"""Independent SM120 BF16 block-sparse attention using warp Tensor Cores.

Four MMA warps consume 16-row K/V tiles from a dedicated producer.
The online softmax spans every selected logical tile.
"""
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.cute.runtime import from_dlpack
from cuda.bindings import driver as cuda

from ._cute_warp_pipeline import acquire_slot, arrive_warp, init_ring, wait_slot


@cute.jit
def _max4(value):
    value = cute.arch.fmax(value, cute.arch.shuffle_sync_bfly(value, 1))
    return cute.arch.fmax(value, cute.arch.shuffle_sync_bfly(value, 2))


@cute.jit
def _sum4(value):
    value += cute.arch.shuffle_sync_bfly(value, 1)
    return value + cute.arch.shuffle_sync_bfly(value, 2)


@cute.jit
def _mma_one(mma, output, a, b):
    out = cute.append_ones(output, up_to_rank=3)
    cute.gemm(mma, out, cute.append_ones(a, up_to_rank=3), cute.append_ones(b, up_to_rank=3), out)


@cute.jit
def _stage_kv(K, V, s_k, s_v, tx, first, length, async_copy):
    # A dedicated warp copies 16-byte vectors into padded shared rows.
    col = tx % 16 * 8
    layout = cute.make_layout(8)
    for i in range(8):
        row = tx // 16 + i * 2
        dst = cute.assume(row * 136 + col, divby=8)
        dst_k = cute.make_tensor(s_k.iterator + dst, layout)
        dst_v = cute.make_tensor(s_v.iterator + dst, layout)
        if row < length:
            src = cute.assume((first + row) * 128 + col, divby=8)
            cute.copy(async_copy, cute.make_tensor(K.iterator + src, layout), dst_k)
            cute.copy(async_copy, cute.make_tensor(V.iterator + src, layout), dst_v)
        else:
            dst_k.fill(0)
            dst_v.fill(0)


@cute.jit
def _key_tile(M, visit, bh, logical_q, NQ: cutlass.Constexpr, NK: cutlass.Constexpr,
              PREFIX: cutlass.Constexpr, SLOTS: cutlass.Constexpr, REGION: cutlass.Constexpr):
    logical_k = visit
    if cutlass.const_expr(REGION == 2):
        if visit >= PREFIX:
            logical_k = M[(bh * NQ + logical_q) * SLOTS + visit - PREFIX]
    return logical_k, logical_k >= 0 and logical_k < NK


@cute.kernel
def _attention_tc(Q: cute.Tensor, K: cute.Tensor, V: cute.Tensor, M: cute.Tensor,
                  KL: cute.Tensor, QL: cute.Tensor, O: cute.Tensor, mma: cute.TiledMma,
                  SQ: cutlass.Constexpr, SK: cutlass.Constexpr, NQ: cutlass.Constexpr,
                  NK: cutlass.Constexpr, LOG2_SCALE: cutlass.Constexpr,
                  PREFIX: cutlass.Constexpr, SLOTS: cutlass.Constexpr, REGION: cutlass.Constexpr,
                  TILE: cutlass.Constexpr):
    tx, _, _ = cute.arch.thread_idx()
    tile_q, bh, _ = cute.arch.block_idx()
    lane = tx % 32
    warp_row = tx // 32 * 16
    first_q = tile_q * 64
    if cutlass.const_expr(REGION == 2):
        first_q += PREFIX * TILE
    logical_q = first_q // TILE
    query_end = logical_q * TILE + QL[logical_q]
    if query_end > SQ:
        query_end = SQ
    thread_mma = mma.get_slice(lane)
    a_coord = thread_mma.partition_A(cute.make_identity_tensor((16, 16)))
    b_coord = thread_mma.partition_B(cute.make_identity_tensor((8, 16)))
    c_coord = thread_mma.partition_C(cute.make_identity_tensor((16, 8)))

    shared = cutlass.memory.SmemAllocator()
    stages = 2
    kv_layout = cute.make_layout((16, 128, stages), stride=(136, 1, 16 * 136))
    k_stages = shared.allocate_tensor(K.element_type, kv_layout, byte_alignment=16)
    v_stages = shared.allocate_tensor(cutlass.BFloat16, kv_layout, byte_alignment=16)
    async_copy = cute.make_copy_atom(cpasync.CopyG2SOp(), cutlass.BFloat16, num_bits_per_copy=128)
    p_layout = cute.make_layout((64, 16), stride=(24, 1))
    s_p = shared.allocate_tensor(cutlass.BFloat16, p_layout, byte_alignment=16)
    s_p_residual = shared.allocate_tensor(cutlass.BFloat16, p_layout, byte_alignment=16)
    compute_threads = 128
    ready, free = init_ring(shared, tx, stages, 1, 4)
    visits = NK
    if cutlass.const_expr(REGION == 2):
        visits = PREFIX + SLOTS
    if tx < compute_threads:
        accumulator = cute.make_rmem_tensor((4, 16), cutlass.Float32)
        accumulator.fill(0.0)
        maximum_0 = cutlass.Float32(float('-inf'))
        maximum_1 = cutlass.Float32(float('-inf'))
        denom_0 = cutlass.Float32(0)
        denom_1 = cutlass.Float32(0)

        # The branch is CTA-uniform. A fully padded query half never touches K/V.
        if first_q < query_end:
            q_fragment = cute.make_rmem_tensor((8, 8), cutlass.BFloat16)
            for depth in cutlass.range_constexpr(8):
                for j in cutlass.range_constexpr(8):
                    row = first_q + warp_row + a_coord[j][0]
                    q_fragment[j, depth] = cutlass.BFloat16(0)
                    if row < query_end:
                        q_fragment[j, depth] = Q[(bh * SQ + row) * 128 + depth * 16 + a_coord[j][1]]

            stage = cutlass.Int32(0)
            for visit in range(visits):
                logical_k, selected = _key_tile(M, visit, bh, logical_q, NQ, NK, PREFIX, SLOTS, REGION)
                if selected:
                    full_length = KL[logical_k]
                    first_k = bh * SK + logical_k * TILE
                    for half in range(TILE // 16):
                        length = full_length - half * 16
                        if length > 0:
                            wait_slot(ready, stage, stages)
                            s_k = k_stages[None, None, stage % stages]
                            s_v = v_stages[None, None, stage % stages]

                            for stripe in range(16 // 16):
                                key_row = stripe * 16
                                if key_row < length:
                                    scores = cute.make_rmem_tensor((4, 2), cutlass.Float32)
                                    scores.fill(0.0)
                                    b_fragment = cute.make_rmem_tensor((4,), cutlass.BFloat16)
                                    for n in cutlass.range_constexpr(2):
                                        for depth in cutlass.range_constexpr(8):
                                            for j in cutlass.range_constexpr(4):
                                                b_fragment[j] = s_k[key_row + n * 8 + b_coord[j][0],
                                                                    depth * 16 + b_coord[j][1]]
                                            _mma_one(mma, scores[None, n], q_fragment[None, depth], b_fragment)
                                        for j in cutlass.range_constexpr(4):
                                            score = cutlass.Float32(float('-inf'))
                                            if key_row + n * 8 + c_coord[j][1] < length:
                                                score = scores[j, n] * LOG2_SCALE
                                            scores[j, n] = score

                                    # C registers 0/1 share one query and 2/3 share
                                    # another. Each query is spread across 4 lanes.
                                    high_0 = cute.arch.fmax(cute.arch.fmax(scores[0, 0], scores[1, 0]),
                                                           cute.arch.fmax(scores[0, 1], scores[1, 1]))
                                    high_1 = cute.arch.fmax(cute.arch.fmax(scores[2, 0], scores[3, 0]),
                                                           cute.arch.fmax(scores[2, 1], scores[3, 1]))
                                    next_0 = cute.arch.fmax(maximum_0, _max4(high_0))
                                    next_1 = cute.arch.fmax(maximum_1, _max4(high_1))
                                    rescale_0 = cute.math.exp2(maximum_0 - next_0, fastmath=True)
                                    rescale_1 = cute.math.exp2(maximum_1 - next_1, fastmath=True)
                                    total_0 = cutlass.Float32(0)
                                    total_1 = cutlass.Float32(0)
                                    for n in cutlass.range_constexpr(2):
                                        for j in cutlass.range_constexpr(4):
                                            peak = next_0 if j < 2 else next_1
                                            probability = cute.math.exp2(scores[j, n] - peak, fastmath=True)
                                            row = warp_row + c_coord[j][0]
                                            col = n * 8 + c_coord[j][1]
                                            rounded = probability.to(cutlass.BFloat16)
                                            s_p[row, col] = rounded
                                            # Recover most of the FP32 probability's lost
                                            # precision using a second BF16 PV product.
                                            s_p_residual[row, col] = (probability - rounded.to(cutlass.Float32)).to(cutlass.BFloat16)
                                            if cutlass.const_expr(j < 2):
                                                total_0 += probability
                                            else:
                                                total_1 += probability
                                    denom_0 = denom_0 * rescale_0 + _sum4(total_0)
                                    denom_1 = denom_1 * rescale_1 + _sum4(total_1)
                                    maximum_0, maximum_1 = next_0, next_1

                                    cute.arch.sync_warp()
                                    p_fragment = cute.make_rmem_tensor((8,), cutlass.BFloat16)
                                    residual_fragment = cute.make_rmem_tensor((8,), cutlass.BFloat16)
                                    for j in cutlass.range_constexpr(8):
                                        row = warp_row + a_coord[j][0]
                                        col = a_coord[j][1]
                                        p_fragment[j] = s_p[row, col]
                                        residual_fragment[j] = s_p_residual[row, col]
                                    for n in cutlass.range_constexpr(16):
                                        for j in cutlass.range_constexpr(4):
                                            factor = rescale_0 if j < 2 else rescale_1
                                            accumulator[j, n] *= factor
                                            b_fragment[j] = s_v[key_row + b_coord[j][1], n * 8 + b_coord[j][0]]
                                        _mma_one(mma, accumulator[None, n], p_fragment, b_fragment)
                                        _mma_one(mma, accumulator[None, n], residual_fragment, b_fragment)
                                    cute.arch.sync_warp()
                            arrive_warp(free, lane, stage, stages)
                            stage += 1

        for n in cutlass.range_constexpr(16):
            for j in cutlass.range_constexpr(4):
                row = first_q + warp_row + c_coord[j][0]
                if row < SQ:
                    denom = denom_0 if j < 2 else denom_1
                    result = cutlass.Float32(0)
                    if row < query_end and denom > 0:
                        result = accumulator[j, n] / denom
                    O[(bh * SQ + row) * 128 + n * 8 + c_coord[j][1]] = result.to(cutlass.BFloat16)
    else:
        if first_q < query_end:
            stage = cutlass.Int32(0)
            for visit in range(visits):
                logical_k, selected = _key_tile(M, visit, bh, logical_q, NQ, NK, PREFIX, SLOTS, REGION)
                if selected:
                    full_length = KL[logical_k]
                    for half in range(TILE // 16):
                        length = full_length - half * 16
                        if length > 0:
                            acquire_slot(free, stage, stages)
                            slot = stage % stages
                            first_k = bh * SK + logical_k * TILE + half * 16
                            _stage_kv(K, V, k_stages[None, None, slot], v_stages[None, None, slot],
                                      tx - compute_threads, first_k, length, async_copy)
                            cute.arch.cp_async_commit_group()
                            cute.arch.cp_async_wait_group(0)
                            arrive_warp(ready, lane, stage, stages)
                            stage += 1



@cute.jit
def _run(Q, K, V, M, KL, QL, O, SQ: cutlass.Constexpr, SK: cutlass.Constexpr,
         NQ: cutlass.Constexpr, NK: cutlass.Constexpr, BH: cutlass.Constexpr,
         LOG2_SCALE: cutlass.Constexpr, PREFIX: cutlass.Constexpr, SLOTS: cutlass.Constexpr,
         REGION: cutlass.Constexpr, QUERY_ROWS: cutlass.Constexpr, TILE: cutlass.Constexpr,
         stream: cuda.CUstream):
    mma = cute.make_tiled_mma(warp.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (16, 8, 16)))
    _attention_tc(Q, K, V, M, KL, QL, O, mma, SQ, SK, NQ, NK, LOG2_SCALE, PREFIX, SLOTS, REGION, TILE).launch(
        grid=((QUERY_ROWS + 63) // 64, BH, 1), block=(160, 1, 1), stream=stream)


_CACHE = {}


def launch(q, k, v, indices, lengths, query_lengths, scale, dense_prefix, tile_size):
    from math import e, log2

    b, h, sq, _ = q.shape
    sk = k.shape[2]
    out = torch.empty_like(q)
    if k.data_ptr() % 16 or v.data_ptr() % 16:
        raise ValueError('Vector K/V loads require 16-byte-aligned inputs')
    tensors = (q, k, v, indices if indices.numel() else lengths, lengths, query_lengths, out)
    views = [from_dlpack(t.detach().view(-1), assumed_align=16 if i in (1, 2) else t.element_size())
             for i, t in enumerate(tensors)]
    stream = cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
    constants = (sq, sk, indices.shape[-2], (sk + tile_size - 1) // tile_size, b * h,
                 scale * log2(e), dense_prefix, indices.shape[-1])
    prefix_rows = min(sq, dense_prefix * tile_size)
    for region, rows in ((1, prefix_rows), (2, sq - prefix_rows)):
        if rows:
            params = (*constants, region, rows, tile_size)
            key = (q.device.index, params)
            compiled = _CACHE.get(key)
            if compiled is None:
                compiled = _CACHE[key] = cute.compile(_run, *views, *params, stream)
            compiled(*views, stream)
    return out
