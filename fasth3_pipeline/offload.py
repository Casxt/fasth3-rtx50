"""Inference-only grouped weight streaming; no CUDA API is touched on CPU.

Stores expose tensor/shape/dtype/validate_keys. Optional synchronous read_into
is preferred to avoid whole-tensor CPU copies/aliases. The caller owns the store.
Root forward is a transaction; for VAE encode/decode use ``with execution()``.
Groups may nest: the most-specific root owns each tensor, while active parent
roots remain resident until their forward returns (including direct qkv use).
Never combine with Accelerate, leaf streaming hooks, training or model.to().
"""
from contextlib import contextmanager
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import get_ident
from time import perf_counter
import torch
from torch import nn

GIB = 1024**3
MIB = 1024**2


class _PrimeBudgetRefusal(MemoryError):
    """Internal budget-only signal, never used for storage/copy errors."""


class _ReadAhead:
    """Bounded CPU-only producer; the caller owns pin slots and CUDA events."""

    def __init__(self, store, pin, stats):
        self.store, self.pin, self.stats = store, pin, stats
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='fasth3-read')
        self.events = [None] * len(pin)
        self.pending = deque()
        self.group = None

    def _read(self, slot, items):
        start = perf_counter()
        end = 0
        for key, offset, size, position in items:
            if position > end:
                self.pin[slot][end:position].zero_()
            copied = self.store.read_into(key, self.pin[slot][position:position + size], byte_offset=offset)
            if type(copied) is not int or copied != size:
                raise IOError(f'Short read_into for {key}: {copied!r} != {size}')
            end = position + size
        return perf_counter() - start

    def _submit(self, slot):
        batch = next(self.plan, None)
        if batch is None:
            return
        if self.events[slot] is not None:
            start = perf_counter()
            self.events[slot].synchronize()
            self.stats['host_wait_seconds'] += perf_counter() - start
            self.events[slot] = None
        begin, items = batch
        self.pending.append((slot, begin, items, self.pool.submit(self._read, slot, items)))

    @staticmethod
    def _batches(entries, offsets, chunk_bytes):
        items, begin = [], 0
        for (_, _, _, key, _, _, size), address in zip(entries, offsets):
            offset = 0
            while offset < size:
                position = address + offset - begin
                if position >= chunk_bytes:
                    if items:
                        yield begin, items
                    begin = address + offset
                    items, position = [], 0
                count = min(chunk_bytes - position, size - offset)
                items.append((key, offset, count, position))
                offset += count
        if items:
            yield begin, items

    def start(self, group, entries, offsets):
        if self.group == group:
            return
        self.drain()
        self.group = group
        self.plan = self._batches(entries, offsets, self.pin[0].numel())
        for slot in range(len(self.pin)):
            self._submit(slot)

    def transfer(self, arena, stream, pending_events):
        while self.pending:
            slot, offset, items, future = self.pending.popleft()
            start = perf_counter()
            try:
                self.stats['read_into_seconds'] += future.result()
            finally:
                self.stats['readahead_wait_seconds'] += perf_counter() - start
            _, _, size, position = items[-1]
            count = position + size
            begin, finish = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record(stream)
            arena[offset:offset + count].copy_(self.pin[slot][:count], non_blocking=True)
            finish.record(stream)
            self.events[slot] = finish
            pending_events.append((begin, finish, 'h2d_gpu_ms'))
            self.stats['read_into_calls'] += len(items)
            self.stats['read_into_bytes'] += sum(size for _, _, size, _ in items)
            self.stats['bytes_h2d'] += count
            self.stats['h2d_chunks'] += 1
            self._submit(slot)

    def drain(self):
        pending = tuple(self.pending)
        self.pending.clear()
        self.group = None
        for _, _, _, future in pending:
            future.cancel()
        error = None
        for _, _, _, future in pending:
            if not future.cancelled():
                try:
                    future.result()
                except Exception as exc:
                    if error is None:
                        error = exc
        if error is not None:
            raise error

    def close(self):
        try:
            self.drain()
        finally:
            self.pool.shutdown(wait=True, cancel_futures=True)


@dataclass(frozen=True)
class WeightGroup:
    name: str
    roots: tuple[str, ...]


def dit_groups(model):
    """Released H3: input, refiners, time projection, main blocks, both heads."""
    candidates = [WeightGroup("input", ("proj_in", "audio_proj_in", "context_embedder")),
                  WeightGroup("refiners", ("token_refiner",)),
                  WeightGroup("time", ("time_proj", "time_embedder"))]
    candidates += [WeightGroup(f"block-{i}", (f"transformer_blocks.{i}",))
                   for i in range(len(model.transformer_blocks))]
    candidates.append(WeightGroup("output", ("norm_out", "proj_out", "audio_proj_out")))
    return candidates


def vae_groups(model):
    """Atomic attention/residual blocks plus parent-owned tensors and projections.

    An attention subtree is never split: processors may read qkv.weight directly.
    FP32 norms retain their checkpoint dtype. Legacy weight_norm prehooks run
    after group loading; transient derived weights are cleared on group release.
    """
    atomic = {"MiniMaxH3VideoResnetBlock3d", "MiniMaxH3VideoDownBlock3d",
              "MiniMaxH3VideoTransformerBlock", "MiniMaxH3AudioCausalAttention",
              "MiniMaxH3AudioResidualUnit", "MiniMaxH3AudioEncoderBlock",
              "MiniMaxH3AudioAMPBlock", "MiniMaxH3AudioAttnProjection"}
    roots = [""]
    for path, module in model.named_modules():
        if path and ("." not in path or type(module).__name__ in atomic):
            if not any(path.startswith(p+".") and type(model.get_submodule(p)).__name__ in atomic
                       for p in roots if p):
                roots.append(path)
    return [WeightGroup(path or "root", (path,)) for path in roots]


class BlockStreamedWeights:
    """Synchronous groups or one-group-ahead H2D prefetch with bounded pinning.

    ``groups`` is a sequence of WeightGroup (or module path strings). Every
    persistent tensor must have exactly one longest matching root. Shared/tied
    persistent tensors are rejected rather than silently untied. Nested roots
    are supported; the scheduler is thread-affine and recursive/concurrent root
    forwards are explicitly rejected. ``prefetch_depth=1`` predicts the next group in this supplied order; dynamic
    execution remains correct when the prediction is wrong. At most ONE idle
    prefetched group is retained; active ancestors are separately budgeted.
    CPU mode simulates residency and prediction, never allocates pinned memory.
    """
    def __init__(self, model, store, device, guard=lambda phase: None, *, groups,
                 prefetch_depth=0, max_gpu_bytes=10*GIB, reserve_bytes=256*MIB,
                 max_resident_bytes=None, max_pinned_bytes=64*MIB, cpu_readahead=False,
                 tensor_guard=None, guard_each_tensor=True, resident_cache_bytes=0):
        if prefetch_depth not in (0, 1):
            raise ValueError("prefetch_depth must be 0 or 1")
        if not 0 < max_gpu_bytes <= 10*GIB or reserve_bytes < 256*MIB:
            raise ValueError("GPU budget must be <=10GiB with >=256MiB reserve")
        if max_pinned_bytes <= 0:
            raise ValueError("Pinned pool must have a positive byte bound")
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("Only CPU and CUDA streaming are supported")
        self.model, self.store, self.guard = model, store, guard
        self.tensor_guard = guard if tensor_guard is None else tensor_guard
        self.guard_each_tensor = guard_each_tensor
        if cpu_readahead and (self.device.type != 'cuda' or prefetch_depth or max_pinned_bytes % 4 or
                              not callable(getattr(store, 'read_into', None))):
            raise ValueError('CPU read-ahead requires CUDA block streaming and a read_into store')
        self.cpu_readahead = cpu_readahead
        self.reader = None
        self.groups = [WeightGroup(g, (g,)) if isinstance(g, str) else g for g in groups]
        if not self.groups or len({g.name for g in self.groups}) != len(self.groups):
            raise ValueError("Group names must be nonempty and unique")
        roots = [(p, i) for i, g in enumerate(self.groups) for p in g.roots]
        self.root_group = next((i for p, i in roots if p == ""), None)
        if len({p for p, _ in roots}) != len(roots):
            raise ValueError("A trigger module may belong to only one group")
        for p, _ in roots:
            model.get_submodule(p)  # Validate before modifying any tensor.
        store.validate_keys(model)
        self.entries = [[] for _ in self.groups]
        seen = set()
        for path, module in model.named_modules():
            for buffer, mapping in ((False, module._parameters), (True, module._buffers)):
                for name, value in mapping.items():
                    if value is None or (buffer and name in module._non_persistent_buffers_set):
                        continue
                    if id(value) in seen:
                        raise ValueError("Tied tensors require an explicit alias-aware scheduler")
                    seen.add(id(value))
                    key = f"{path}.{name}" if path else name
                    choices = [(len(p), i) for p, i in roots if not p or path == p or path.startswith(p+".")]
                    if not choices:
                        raise ValueError(f"No group owns persistent tensor {key}")
                    _, group = max(choices)
                    dtype = store.dtype(key)
                    shape = tuple(store.shape(key))
                    size = value.numel()*torch.empty((), dtype=dtype).element_size()
                    self.entries[group].append((module, name, buffer, key, shape, dtype, size))
        # Non-persistent tensors must already contain initialized values, not meta.
        self.buffers = [(module, name, value) for module in model.modules()
                        for name, value in module._buffers.items()
                        if value is not None and name in module._non_persistent_buffers_set]
        if any(value.is_meta for _, _, value in self.buffers):
            raise ValueError("Initialize non-persistent buffers outside meta before streaming")
        self.derived = [[] for _ in self.groups]
        for i, entries in enumerate(self.entries):
            for module, name, _, _, _, _, _ in entries:
                if name == "weight_g" and hasattr(module, "weight_v"):
                    self.derived[i].append(module)
        self.sizes = [sum(e[-1] for e in entries) for entries in self.entries]
        if cpu_readahead:
            self.offsets = []
            for index, entries in enumerate(self.entries):
                offsets, end = [], 0
                for *_, size in entries:
                    # Typed views retain allocator alignment, including for TMA.
                    offset = (end + 255) // 256 * 256
                    offsets.append(offset)
                    end = offset + size
                self.offsets.append(offsets)
                self.sizes[index] = end
        self.prefetch_depth = prefetch_depth
        self._prefetch_omitted = frozenset()
        self.max_gpu_bytes, self.reserve_bytes = max_gpu_bytes, reserve_bytes
        self.max_resident_bytes = max_resident_bytes if max_resident_bytes is not None else max_gpu_bytes-reserve_bytes
        if not 0 < self.max_resident_bytes <= max_gpu_bytes:
            raise ValueError("Invalid resident byte bound")
        if max(self.sizes, default=0) > self.max_resident_bytes:
            raise MemoryError("A group exceeds the resident byte bound; split its roots")
        if (not 0 <= resident_cache_bytes <= self.max_resident_bytes or
                resident_cache_bytes and (prefetch_depth or cpu_readahead)):
            raise ValueError("Resident cache must fit the budget and use demand loading")
        self.cache_limit = int(resident_cache_bytes)
        self.cached = {}
        self.cached_bytes = 0
        self.max_pinned_bytes = int(max_pinned_bytes)
        self.handles, self.loaded, self.ready = [], {}, {}
        self.active = [0]*len(self.groups)
        self.current = self.prefetched = self.primed_group = None
        self.running = self.closed = self.in_forward = self.priming_attempt = False
        self.owner_thread = None
        self.construction_thread = get_ident()
        self.rejected_posts = 0
        self.copy_stream = self.pin = self.pin_event = None
        self.pending_events = []
        self.stats = dict(group_loads=0, prefetch_loads=0, prefetch_hits=0,
                          prefetch_skips=0, bytes_read=0, bytes_h2d=0,
                          resident_bytes=0, resident_peak_bytes=0,
                          pinned_bytes=0, pinned_peak_bytes=0, runs=0,
                          store_read_seconds=0., pinned_copy_seconds=0.,
                          read_into_seconds=0., read_into_bytes=0, read_into_calls=0,
                          prime_calls=0, prime_loads=0, prime_hits=0,
                          prime_skips=0, prime_cancels=0,
                          readahead_wait_seconds=0., readahead_early_groups=0,
                          guard_seconds=0., guard_calls=0,
                          host_wait_seconds=0., h2d_gpu_ms=0., h2d_chunks=0,
                          compute_gpu_ms=0., cache_hits=0, cache_evictions=0, cache_peak_bytes=0)
        self.group_calls = {}
        try:
            for entries in self.entries:
                for e in entries:
                    self._meta(e)
            for modules in self.derived:
                for module in modules:
                    module.weight = torch.empty_like(module.weight, device="meta")
            for path, index in roots:
                module = model.get_submodule(path)
                calls = []
                def before(mod, args, index=index, calls=calls):
                    calls.append(False)
                    if torch.is_grad_enabled() and _grad_inputs(args):
                        raise RuntimeError("Block streaming is inference-only; disable autograd")
                    self._enter_group(index)
                    self.active[index] += 1
                    calls[-1] = True
                    if self.device.type == "cuda":
                        event = torch.cuda.Event(enable_timing=True)
                        event.record(torch.cuda.current_stream(self.device))
                        self.group_calls.setdefault(id(calls), []).append(event)
                def after(mod, args, output, index=index, calls=calls):
                    if get_ident() != self.construction_thread or (mod is self.model and self.rejected_posts):
                        return
                    entered = calls.pop() if calls else False
                    if entered:
                        self.active[index] -= 1
                        if self.device.type == "cuda":
                            start = self.group_calls[id(calls)].pop()
                            end = torch.cuda.Event(enable_timing=True)
                            end.record(torch.cuda.current_stream(self.device))
                            self.pending_events.append((start, end, "compute_gpu_ms"))
                        if index != self.current and not self.active[index] and index not in self.cached:
                            self._release(index)
                        elif index == self.current:
                            # Compute has now been enqueued. Feeding the next
                            # group here overlaps it instead of delaying launch.
                            self._prefetch(index)
                self.handles.append(module.register_forward_pre_hook(before, prepend=True))
                self.handles.append(module.register_forward_hook(after, always_call=True))
            self.handles.append(model.register_forward_pre_hook(self._root_pre, prepend=True))
            self.handles.append(model.register_forward_hook(self._root_post, always_call=True))
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _meta(entry):
        module, name, buffer, _, shape, dtype, _ = entry
        empty = torch.empty(shape, device="meta", dtype=dtype)
        (module._buffers if buffer else module._parameters)[name] = empty if buffer else nn.Parameter(empty, requires_grad=False)

    def _cuda_capacity(self, size):
        free, _ = torch.cuda.mem_get_info(self.device)
        allocated = torch.cuda.memory_allocated(self.device)
        reserved = torch.cuda.memory_reserved(self.device)
        memory = torch.cuda.memory_stats(self.device)
        # Native allocator active bytes include blocks awaiting record_stream
        # completion, unlike allocated bytes. Unknown backends conservatively
        # treat ALL reserved bytes as active; cached bytes are never called free.
        active = max(allocated, memory.get("active_bytes.all.current", reserved))
        self.stats["last_cuda_capacity"] = dict(required_bytes=size, free_bytes=free,
                                                allocated_bytes=allocated,
                                                active_bytes=active, reserved_bytes=reserved)
        return (free >= size+self.reserve_bytes and reserved <= self.max_gpu_bytes
                and active+size <= self.max_gpu_bytes)

    def _check_idle(self):
        if get_ident() != self.construction_thread:
            raise RuntimeError("Block scheduler is thread-affine; concurrent calls are unsupported")
        if self.closed:
            raise RuntimeError("Block scheduler is closed")
        if self.running or self.in_forward or self.priming_attempt:
            raise RuntimeError("Priming/cancellation requires an idle scheduler")

    def _guard(self, phase, check=None):
        started = perf_counter()
        try:
            (self.guard if check is None else check)(phase)
        finally:
            self.stats['guard_seconds'] += perf_counter() - started
            self.stats['guard_calls'] += 1

    def _drop_pin(self):
        # Caller must first synchronize the copy stream (_end does so).
        try:
            if self.reader is not None:
                self.reader.close()
        finally:
            self.reader = self.pin = self.pin_event = None
            self.stats["pinned_bytes"] = 0

    def prime_first(self):
        """Optionally stage only the first NONEMPTY execution group while idle.

        Intended for an external VAE posthook. Never marks this scheduler as a
        running forward: its next root forward owns the transaction and cleanup.
        Return False for a budget refusal/empty model, but propagate actual I/O
        or copy errors after cleaning up. Repeated calls are idempotent while
        primed. GPU readiness and the owned pin slot remain alive until consumed
        or explicitly cancelled/closed. This is not a promise of physical overlap.
        """
        self._check_idle()
        self.stats["prime_calls"] += 1
        if self.primed_group is not None:
            return True
        if self.loaded:
            raise RuntimeError("Idle scheduler contains unowned resident groups")
        first = next((i for i, entries in enumerate(self.entries) if entries), None)
        if first is None:
            self.stats["prime_skips"] += 1
            return False
        try:
            self._guard("block prime start")
        except MemoryError:
            self.stats["prime_skips"] += 1
            return False
        self.priming_attempt = True
        try:
            if self.device.type == "cuda" and self.copy_stream is None:
                self.copy_stream = torch.cuda.Stream(device=self.device)
            if not self._load(first, optional_budget=True):
                self._end()
                self._drop_pin()
                self.stats["prime_skips"] += 1
                return False
            self.primed_group = first
            self.stats["prime_loads"] += 1
            return True
        except _PrimeBudgetRefusal:
            try:
                self._end()
            finally:
                self._drop_pin()
            self.stats["prime_skips"] += 1
            return False
        except BaseException:
            try:
                self._end()
            finally:
                self._drop_pin()
            raise
        finally:
            self.priming_attempt = False

    def cancel_prime(self):
        """Cancel idle priming, including pending DMA; False if nothing primed."""
        self._check_idle()
        if self.primed_group is None:
            return False
        try:
            self._end()
        finally:
            self._drop_pin()
        self.stats["prime_cancels"] += 1
        return True

    def _begin(self):
        if get_ident() != self.construction_thread:
            raise RuntimeError("Block scheduler is thread-affine; concurrent calls are unsupported")
        if self.closed:
            raise RuntimeError("Block scheduler is closed")
        if self.running:
            raise RuntimeError("Reentrant/concurrent execution is not supported")
        self.running, self.owner_thread = True, get_ident()
        self.stats["runs"] += 1
        try:
            self._guard("block execution start")
            if self.device.type == "cuda" and self.copy_stream is None:
                self.copy_stream = torch.cuda.Stream(device=self.device)
            for module, name, value in self.buffers:
                self._guard(f"block buffer:{name}")
                if self.device.type == "cuda" and not self._cuda_capacity(value.numel()*value.element_size()):
                    raise MemoryError("GPU headroom insufficient for nonpersistent buffer")
                module._buffers[name] = value.to(self.device)
        except BaseException:
            self._end()
            raise

    def _root_pre(self, module, args):
        if get_ident() != self.construction_thread:
            raise RuntimeError("Block scheduler is thread-affine; concurrent calls are unsupported")
        if self.in_forward or self.priming_attempt or (self.running and get_ident() != self.owner_thread):
            self.rejected_posts += 1
            raise RuntimeError("Reentrant/concurrent root forward is not supported")
        self.in_forward = True
        self.owns_run = not self.running
        if self.owns_run:
            self._begin()

    def _root_post(self, module, args, output):
        if get_ident() != self.construction_thread:
            return
        if self.rejected_posts:
            self.rejected_posts -= 1
            return
        if self.in_forward:
            try:
                if self.owns_run:
                    self._end()
            finally:
                self.in_forward = False

    @contextmanager
    def execution(self):
        """Alternate-entry transaction, including root-owned persistent state.

        encode/decode need not call root.forward, so its hook cannot provide
        liveness. Hold its group explicitly until the entire context returns.
        """
        self._begin()
        try:
            if self.root_group is not None:
                self._enter_group(self.root_group)
                self.active[self.root_group] += 1
            yield self
        finally:
            self._end()

    def _ensure_pin(self):
        if self.pin is None:
            try:
                self._guard("allocate bounded pinned pool")
            except MemoryError as error:
                if self.priming_attempt:
                    raise _PrimeBudgetRefusal(str(error)) from error
                raise
            try:
                self.pin = torch.empty(self.max_pinned_bytes, dtype=torch.uint8, pin_memory=True)
            except torch.OutOfMemoryError as error:
                if self.priming_attempt:
                    raise _PrimeBudgetRefusal(str(error)) from error
                raise
            self.stats["pinned_bytes"] = self.pin.numel()
            self.stats["pinned_peak_bytes"] = max(self.stats["pinned_peak_bytes"], self.pin.numel())

    def _start_readahead(self, index):
        if self.reader is None:
            self._ensure_pin()
            # Split the bounded host pin pool into four independent slots that
            # let the producer read while a previous slot is consumed by DMA.
            self.reader = _ReadAhead(self.store, self.pin.split(self.max_pinned_bytes // 4), self.stats)
        self.reader.start(index, self.entries[index], self.offsets[index])

    def _copy(self, source, destination):
        if self.device.type == "cpu":
            destination.copy_(source)
            return
        self._ensure_pin()
        src, dst = source.reshape(-1).view(torch.uint8), destination.reshape(-1).view(torch.uint8)
        for offset in range(0, src.numel(), self.max_pinned_bytes):
            if self.pin_event is not None:
                start = perf_counter()
                self.pin_event.synchronize()  # Never overwrite a DMA source in flight.
                self.stats["host_wait_seconds"] += perf_counter()-start
            end = min(offset+self.max_pinned_bytes, src.numel())
            chunk = self.pin[:end-offset]
            start = perf_counter()
            chunk.copy_(src[offset:end])  # Synchronous CPU mmap -> pinned ownership.
            self.stats["pinned_copy_seconds"] += perf_counter()-start
            begin, finish = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record(self.copy_stream)
            dst[offset:end].copy_(chunk, non_blocking=True)
            finish.record(self.copy_stream)
            self.pin_event = finish
            self.pending_events.append((begin, finish, "h2d_gpu_ms"))
            self.stats["bytes_h2d"] += end-offset
            self.stats["h2d_chunks"] += 1

    def _copy_from_store(self, key, destination):
        """Fill owned storage without requesting a complete CPU tensor alias.

        read_into is a synchronous CPU byte-copy contract for BOTH storage
        backends; it must never return an alias or enqueue a deferred write.
        The same owned pin slot remains alive until its last H2D event completes.
        """
        target = destination.reshape(-1).view(torch.uint8)
        cuda = self.device.type == "cuda"
        if cuda and target.numel():
            self._ensure_pin()
        for offset in range(0, target.numel(), self.max_pinned_bytes):
            end = min(offset+self.max_pinned_bytes, target.numel())
            if cuda:
                if self.pin_event is not None:
                    start = perf_counter()
                    self.pin_event.synchronize()
                    self.stats["host_wait_seconds"] += perf_counter()-start
                chunk = self.pin[:end-offset]
            else:
                chunk = target[offset:end]
            start = perf_counter()
            self.stats["read_into_calls"] += 1
            try:
                copied = self.store.read_into(key, chunk, byte_offset=offset)
            finally:
                self.stats["read_into_seconds"] += perf_counter()-start
            if type(copied) is not int or copied != chunk.numel():
                raise IOError(f"read_into must return exact integer byte count for {key}: {copied!r} != {chunk.numel()}")
            self.stats["read_into_bytes"] += copied
            if cuda:
                begin, finish = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record(self.copy_stream)
                target[offset:end].copy_(chunk, non_blocking=True)
                finish.record(self.copy_stream)
                self.pin_event = finish
                self.pending_events.append((begin, finish, "h2d_gpu_ms"))
                self.stats["bytes_h2d"] += copied
                self.stats["h2d_chunks"] += 1

    def _load_arena(self, index, size, optional_budget):
        try:
            arena = torch.empty(size, device=self.device, dtype=torch.uint8)
        except torch.OutOfMemoryError:
            if optional_budget:
                return False
            raise
        self.loaded[index].append(arena)
        self.stats['resident_bytes'] += size
        self.stats['resident_peak_bytes'] = max(self.stats['resident_peak_bytes'], self.stats['resident_bytes'])
        try:
            self._guard(f"block arena:{self.groups[index].name}", self.tensor_guard)
        except MemoryError:
            if optional_budget:
                return False
            raise
        self._start_readahead(index)
        self.reader.transfer(arena, self.copy_stream, self.pending_events)
        for (module, name, buffer, _, shape, dtype, nbytes), offset in zip(self.entries[index], self.offsets[index]):
            destination = arena[offset:offset + nbytes].view(dtype).view(shape)
            mapping = module._buffers if buffer else module._parameters
            mapping[name] = destination if buffer else nn.Parameter(destination, requires_grad=False)
            self.stats['bytes_read'] += nbytes
        return True

    def _load(self, index, *, prefetch=False, optional_budget=False):
        if index in self.loaded:
            return True
        size = self.sizes[index]
        if self.stats["resident_bytes"]+size > self.max_resident_bytes:
            if optional_budget:
                return False
            raise MemoryError("Active groups exceed resident byte bound")
        try:
            self._guard(f"block load:{self.groups[index].name}")
        except MemoryError:
            if optional_budget:
                return False
            raise
        if self.device.type == "cuda" and not self._cuda_capacity(size):
            if optional_budget:
                return False
            raise MemoryError("GPU headroom insufficient for requested group")
        self.loaded[index] = []
        try:
            scope = torch.cuda.stream(self.copy_stream) if self.device.type == "cuda" else _null_scope()
            with scope:
                packed = self.cpu_readahead and size > 0
                if packed and not self._load_arena(index, size, optional_budget):
                    self._release(index)
                    return False
                for entry in () if packed else self.entries[index]:
                    module, name, buffer, key, shape, dtype, nbytes = entry
                    if self.guard_each_tensor:
                        try:
                            self._guard(f"block tensor:{key}", self.tensor_guard)
                        except MemoryError:
                            if optional_budget:
                                self._release(index)
                                return False
                            raise
                    try:
                        destination = torch.empty(shape, dtype=dtype, device=self.device)
                    except torch.OutOfMemoryError:
                        # Only allocation OOM is an optional-capacity refusal.
                        # Store/copy exceptions, even MemoryError, propagate below.
                        if optional_budget and self.device.type == "cuda":
                            self._release(index)
                            return False
                        raise
                    self.loaded[index].append(destination)
                    self.stats["resident_bytes"] += nbytes
                    self.stats["resident_peak_bytes"] = max(self.stats["resident_peak_bytes"], self.stats["resident_bytes"])
                    if callable(getattr(self.store, "read_into", None)):
                        self._copy_from_store(key, destination)
                    else:
                        start = perf_counter()
                        source = self.store.tensor(key)
                        self.stats["store_read_seconds"] += perf_counter()-start
                        if source.device.type != "cpu" or tuple(source.shape) != shape or source.dtype != dtype or not source.is_contiguous():
                            raise ValueError(f"Invalid contiguous CPU store tensor: {key}")
                        self._copy(source, destination)
                        del source
                    mapping = module._buffers if buffer else module._parameters
                    mapping[name] = destination if buffer else nn.Parameter(destination, requires_grad=False)
                    self.stats["bytes_read"] += nbytes
                if self.device.type == "cuda":
                    event = torch.cuda.Event()
                    event.record(self.copy_stream)
                    self.ready[index] = event
            self.stats["group_loads"] += 1
            self.stats["prefetch_loads"] += int(prefetch)
            return True
        except BaseException:
            self._release(index)
            raise

    def _release(self, index):
        values = self.loaded.pop(index, None)
        if values is None:
            return
        # CUDA allocator sees compute-stream ownership in _enter_group. The
        # copy stream owns allocation/copies; pending pin events retain sources.
        self.stats["resident_bytes"] -= sum(v.numel()*v.element_size() for v in values)
        self.cached_bytes -= self.cached.pop(index, 0)
        for entry in self.entries[index]:
            self._meta(entry)
        for module in self.derived[index]:
            module.weight = torch.empty_like(module.weight, device="meta")
        self.ready.pop(index, None)
        if self.primed_group == index:
            self.primed_group = None
        if self.prefetched == index:
            self.prefetched = None

    def _enter_group(self, index):
        if not self.running or get_ident() != self.owner_thread:
            raise RuntimeError("Use execution() around non-forward model entry points")
        if self.current != index:
            for other in list(self.loaded):
                # Empty parent/setup triggers are not a wrong execution
                # prediction: retain priming until the first weighted group.
                keep_prime = other == self.primed_group and not self.entries[index]
                if other != index and not self.active[other] and not keep_prime and other not in self.cached:
                    self._release(other)
            if self.primed_group == index:
                self.stats["prime_hits"] += 1
                self.primed_group = None
            if self.prefetched == index:
                self.stats["prefetch_hits"] += 1
                self.prefetched = None
            if index in self.cached:
                self.stats["cache_hits"] += 1
                # Cached weights bypass the per-load guard. Keep checking
                # external CUDA headroom during repeated inference tiles.
                if self.device.type == 'cuda' and self.stats["cache_hits"] % 32 == 0:
                    self._guard(f"cached group:{self.groups[index].name}", self.tensor_guard)
            elif index not in self.loaded and self.cached:
                # Keep the first groups that fit the cache. A recency cache
                # would thrash on every tile when the repeated cycle is larger.
                size = self.sizes[index]
                for other in reversed(tuple(self.cached)):
                    if (self.stats["resident_bytes"] + size <= self.max_resident_bytes and
                            (self.device.type != 'cuda' or self._cuda_capacity(size))):
                        break
                    if not self.active[other]:
                        self._release(other)
                        self.stats["cache_evictions"] += 1
            self._load(index)
            size = self.sizes[index]
            if size and index not in self.cached and self.cached_bytes + size <= self.cache_limit:
                self.cached[index] = size
                self.cached_bytes += size
                self.stats["cache_peak_bytes"] = max(self.stats["cache_peak_bytes"], self.cached_bytes)
            self.current = index
            next_index = index + 1
            if (self.cpu_readahead and next_index < len(self.entries) and
                    self.sizes[next_index] and next_index not in self.loaded):
                self._start_readahead(next_index)
                self.stats['readahead_early_groups'] += 1
        if self.device.type == "cuda":
            compute = torch.cuda.current_stream(self.device)
            compute.wait_event(self.ready[index])
            for value in self.loaded[index]:
                value.record_stream(compute)
    @contextmanager
    def skip_prefetch_groups(self, names):
        """Skip speculative loads only, leaving unexpected demand loads intact."""
        previous = self._prefetch_omitted
        self._prefetch_omitted = frozenset(i for i, group in enumerate(self.groups) if group.name in names)
        try:
            yield
        finally:
            self._prefetch_omitted = previous

    def _prefetch(self, index):
        next_index = index+1
        while next_index in self._prefetch_omitted:
            next_index += 1
        if self.prefetch_depth and self.prefetched is None and next_index < len(self.groups) and next_index not in self.loaded:
            size = self.sizes[next_index]
            possible = self.stats["resident_bytes"]+size <= self.max_resident_bytes
            if self.device.type == "cuda":
                possible = possible and self._cuda_capacity(size)
            if possible:
                # Only a preflight capacity refusal is a skip. Once loading has
                # begun, OOM/I/O/copy errors cancel the transaction and propagate
                # unchanged; never disguise arbitrary RuntimeError as an OOM.
                self._load(next_index, prefetch=True)
                self.prefetched = next_index
            else:
                self.stats["prefetch_skips"] += 1

    def _harvest(self, *, wait=False):
        remaining = []
        for start, end, field in self.pending_events:
            if wait:
                end.synchronize()
            if end.query():
                self.stats[field] += start.elapsed_time(end)
            else:
                remaining.append((start, end, field))
        self.pending_events = remaining

    def _end(self):
        try:
            if self.copy_stream is not None:
                self.copy_stream.synchronize()  # Including prefetched-but-unused copies.
                self._harvest(wait=True)
            if self.reader is not None:
                self.reader.drain()
        finally:
            for index in list(self.loaded):
                self._release(index)
            for module, name, original in self.buffers:
                module._buffers[name] = original
            self.active = [0]*len(self.groups)
            self.current = self.prefetched = self.primed_group = None
            self.running, self.owner_thread = False, None

    def stats_snapshot(self):
        if self.device.type == "cuda":
            self._harvest()
        return {**self.stats, "prefetch_depth": self.prefetch_depth,
                "resident_cache_limit_bytes": self.cache_limit, "cached_bytes": self.cached_bytes,
                "loaded_groups": [self.groups[i].name for i in self.loaded],
                "primed_group": None if self.primed_group is None else self.groups[self.primed_group].name,
                "group_bytes": {g.name: n for g, n in zip(self.groups, self.sizes)},
                "pending_cuda_timings": len(self.pending_events),
                "supply_api": "read_into" if callable(getattr(self.store, "read_into", None)) else "tensor",
                "storage_backend": getattr(self.store, "backend", "external"),
                "io_bytes_basis": "logical payload bytes, not physical disk reads",
                "timing_note": "group compute-stream interval, not summed kernel busy time; may include CPU submission/stream idle; copy/compute overlap and nested scopes are not additive"}

    def close(self):
        """Close idle/primed state; never invalidate a live foreground forward."""
        if self.closed:
            return
        self._check_idle()
        try:
            self._end()
        finally:
            for handle in self.handles:
                handle.remove()
            self.handles.clear()
            try:
                self._drop_pin()
            finally:
                self.copy_stream = None
                self.closed = True


def _grad_inputs(value):
    if isinstance(value, torch.Tensor):
        return value.requires_grad
    if isinstance(value, dict):
        return any(_grad_inputs(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return any(_grad_inputs(v) for v in value)
    return False


@contextmanager
def _null_scope():
    yield
