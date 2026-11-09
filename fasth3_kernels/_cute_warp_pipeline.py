"""CTA-local shared-memory handoff between producer and consumer warps."""
import cutlass
import cutlass.cute as cute


@cute.jit
def init_ring(smem, tx, stages: cutlass.Constexpr,
              producers: cutlass.Constexpr, consumers: cutlass.Constexpr):
    ready = smem.allocate_tensor(cutlass.Int64, cute.make_layout(stages), byte_alignment=8)
    free = smem.allocate_tensor(cutlass.Int64, cute.make_layout(stages), byte_alignment=8)
    if tx == 0:
        for slot in cutlass.range_constexpr(stages):
            cute.arch.mbarrier_init(ready.iterator + slot, producers)
            cute.arch.mbarrier_init(free.iterator + slot, consumers)
        cute.arch.mbarrier_init_fence()
    cute.arch.sync_threads()
    return ready, free


@cute.jit
def acquire_slot(free, index, stages: cutlass.Constexpr):
    if index >= stages:
        cute.arch.mbarrier_wait(free.iterator + index % stages, (index // stages - 1) % 2)


@cute.jit
def wait_slot(ready, index, stages: cutlass.Constexpr):
    cute.arch.mbarrier_wait(ready.iterator + index % stages, index // stages % 2)


@cute.jit
def arrive_warp(barriers, lane, index, stages: cutlass.Constexpr):
    cute.arch.sync_warp()
    if lane == 0:
        cute.arch.mbarrier_arrive(barriers.iterator + index % stages)
