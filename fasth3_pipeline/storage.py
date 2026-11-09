"""Persistent CPU safetensors views and explicitly byte-bounded owned staging.

No implicit CUDA initialization, global state_dict, or fp8 import. Default safe_open
returns read-only-by-contract private-map CPU views. Optional readonly_mmap uses
rb + mmap.ACCESS_READ; file_readinto uses cached OS file I/O without mapping.
Both return owned tensors or fill caller-owned CPU byte storage. Keep owned
buffers alive until H2D completion. Virtual bytes are NOT resident/physical I/O
bytes, and cached file reads and mappings both use the OS page cache.
"""
from collections import OrderedDict
from dataclasses import dataclass
import json
import math
import mmap
import os
from pathlib import Path
import struct
import threading
import time
import torch

INDEX = 'diffusion_pytorch_model.safetensors.index.json'
_TYPES = {'F64': torch.float64, 'F32': torch.float32, 'F16': torch.float16,
          'BF16': torch.bfloat16, 'F8_E4M3': torch.float8_e4m3fn,
          'F8_E5M2': torch.float8_e5m2, 'I64': torch.int64, 'I32': torch.int32,
          'I16': torch.int16, 'I8': torch.int8, 'U8': torch.uint8, 'BOOL': torch.bool}
_SIZES = {'F64': 8, 'F32': 4, 'F16': 2, 'BF16': 2, 'F8_E4M3': 1,
          'F8_E5M2': 1, 'I64': 8, 'I32': 4, 'I16': 2, 'I8': 1, 'U8': 1, 'BOOL': 1}


def _positive(value, name, zero=False):
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if zero else 1):
        raise ValueError(f'{name} must be an integer >= {0 if zero else 1}')
    return value


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON key: {key}')
        result[key] = value
    return result


@dataclass(frozen=True)
class TensorInfo:
    shape: tuple
    dtype: torch.dtype
    nbytes: int
    offset: int  # Absolute file byte offset, not a physical I/O count.


class TensorBatch:
    """Owned CPU staging lease. Release only AFTER all H2D consumers finish.

    Do not retain tensor aliases beyond close(). Closing drops this batch's
    references/accounting; Python aliases cannot be forcibly reclaimed.
    """
    def __init__(self, store, tensors, nbytes):
        self._store, self._tensors, self.nbytes = store, tensors, nbytes
        self.closed = False

    def __getitem__(self, key):
        if self.closed:
            raise RuntimeError('TensorBatch is closed')
        return self._tensors[key]

    def keys(self):
        if self.closed:
            raise RuntimeError('TensorBatch is closed')
        return self._tensors.keys()

    def close(self):
        with self._store._lock:
            if not self.closed:
                self._tensors.clear()
                self._store._stats['staging_bytes'] -= self.nbytes
                self._store._stats['active_batches'] -= 1
                self.closed = True

    def __enter__(self):
        if self.closed:
            raise RuntimeError('TensorBatch is closed')
        return self

    def __exit__(self, *args):
        self.close()


def _copy_bytes(destination, source):
    """Synchronous memcpy between byte memoryviews; isolated for fault injection."""
    destination[:] = source


class _ReadOnlyMapping:
    """Own an rb fd + ACCESS_READ map; never expose a Torch source alias."""
    def __init__(self, path, identity):
        self.file = path.open('rb')
        self.mapping = None
        try:
            stat = os.fstat(self.file.fileno())
            if (stat.st_size, stat.st_mtime_ns, stat.st_ino) != identity:
                raise RuntimeError('Shard changed before readonly mapping')
            self.mapping = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
            self.nbytes = stat.st_size
        except BaseException:
            self.file.close()
            raise

    def __enter__(self):
        return self.mapping

    def __exit__(self, *args):
        try:
            self.mapping.close()
        finally:
            self.file.close()


class PersistentTensorStore:
    """Read-only, thread-safe reader with bounded handles, headers and CPU copies.

    max_cache_bytes controls OPTIONAL owned CPU tensor copies (default zero).
    max_staging_bytes controls all active stage() leases together (default256MiB).
    Defaults: max_open_shards=2, max_mapped_bytes=8GiB for store-managed extents.
    Header/handle eviction is independent of leases. safe_open views own native
    storage references and may outlive eviction (their extents are NOT tracked
    afterwards). The readonly_mmap and file_readinto backends return owned tensors or copy into caller
    buffers; no mapping view escapes. tensor() returned-copy bytes, caller aliases,
    and OS page cache are outside the staging/cache caps. Callers must budget those
    allocations and additionally use the application host/commit guard.
    """
    index_filename = INDEX

    def __init__(self, directory, *, backend='safe_open', max_open_shards=2, max_cache_bytes=0,
                 max_staging_bytes=256 << 20, max_header_bytes=64 << 20,
                 max_index_bytes=8 << 20, max_mapped_bytes=8 << 30):
        if backend not in ('safe_open', 'readonly_mmap', 'file_readinto'):
            raise ValueError('Unknown tensor reader backend')
        self.backend = backend
        self.max_mapped_bytes = _positive(max_mapped_bytes, 'max_mapped_bytes')
        self.max_open_shards = _positive(max_open_shards, 'max_open_shards')
        self.max_cache_bytes = _positive(max_cache_bytes, 'max_cache_bytes', True)
        self.max_staging_bytes = _positive(max_staging_bytes, 'max_staging_bytes')
        self.max_header_bytes = _positive(max_header_bytes, 'max_header_bytes')
        _positive(max_index_bytes, 'max_index_bytes')
        self.directory = Path(directory).resolve(strict=True)
        if not self.directory.is_dir():
            raise ValueError('Store directory is not a directory')
        index = (self.directory / self.index_filename).resolve(strict=True)
        if index.parent != self.directory or index.stat().st_size > max_index_bytes:
            raise ValueError('Unsafe or oversized safetensors index')
        data = json.loads(index.read_text(encoding='utf-8'), object_pairs_hook=_unique)
        if not isinstance(data, dict):
            raise ValueError('Invalid safetensors index object')
        self.weight_map = data.get('weight_map')
        if not isinstance(self.weight_map, dict) or not self.weight_map:
            raise ValueError('Empty/invalid safetensors index')
        self._paths, self._expected = {}, {}
        for key, shard in self.weight_map.items():
            if not isinstance(key, str) or not key or not isinstance(shard, str):
                raise ValueError('Invalid tensor key or shard name')
            if (not shard.endswith('.safetensors') or '/' in shard or '\\' in shard
                    or ':' in shard or Path(shard).name != shard):
                raise ValueError(f'Unsafe shard path: {shard}')
            path = (self.directory / shard).resolve(strict=True)
            if path.parent != self.directory or not path.is_file():
                raise ValueError(f'Missing or unsafe shard path: {shard}')
            self._paths[shard] = path
            self._expected.setdefault(shard, set()).add(key)
        # Capture sizes/mtime to reject ordinary replacement while the store lives.
        # This is NOT an adversarial filesystem TOCTOU/security boundary.
        self._identity = {s: self._signature(p) for s, p in self._paths.items()}
        self._readers, self._headers, self._cache = OrderedDict(), OrderedDict(), OrderedDict()
        self._header_bytes = 0
        self._lock = threading.RLock()
        self.closed = False
        self._stats = dict(shard_opens=0, handle_hits=0, header_reads=0, header_hits=0,
                           tensor_requests=0, logical_tensor_bytes=0, cache_hits=0,
                           cache_bytes=0, staging_bytes=0, peak_staging_bytes=0,
                           active_batches=0, staged_bytes_total=0, staging_seconds=0.0,
                           read_into_calls=0, read_into_bytes=0, read_into_failures=0,
                           mapped_file_bytes=0, peak_mapped_file_bytes=0,
                           mapped_copy_bytes_total=0, owned_tensor_bytes_total=0)

    @staticmethod
    def _signature(path):
        stat = path.stat()
        return stat.st_size, stat.st_mtime_ns, stat.st_ino

    def _check(self):
        if self.closed:
            raise RuntimeError('PersistentTensorStore is closed')

    def _path(self, shard):
        path = self._paths[shard]
        if self._signature(path) != self._identity[shard]:
            raise RuntimeError(f'Shard changed during reader lifetime: {shard}')
        return path

    def _header(self, shard):
        self._check()
        path = self._path(shard)
        if shard in self._headers:
            self._stats['header_hits'] += 1
            self._headers.move_to_end(shard)
            return self._headers[shard][0]
        with path.open('rb') as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ValueError(f'Truncated safetensors header: {shard}')
            length = struct.unpack('<Q', prefix)[0]
            size = self._identity[shard][0]
            if length < 2 or length > self.max_header_bytes or 8 + length > size:
                raise ValueError(f'Invalid/oversized safetensors header: {shard}')
            raw = stream.read(length)
        header = json.loads(raw, object_pairs_hook=_unique)
        if not isinstance(header, dict):
            raise ValueError(f'Invalid header: {shard}')
        entries, ranges = {}, []
        for key, item in header.items():
            if key == '__metadata__':
                if not isinstance(item, dict) or any(not isinstance(k, str) or not isinstance(v, str)
                                                     for k, v in item.items()):
                    raise ValueError(f'Invalid safetensors metadata: {shard}')
                continue
            if not isinstance(item, dict):
                raise ValueError(f'Invalid tensor metadata: {key}')
            shape, name, offsets = item.get('shape'), item.get('dtype'), item.get('data_offsets')
            if (not isinstance(shape, list) or any(type(x) is not int or x < 0 for x in shape)
                    or not isinstance(name, str) or name not in _TYPES
                    or not isinstance(offsets, list) or len(offsets) != 2
                    or any(type(x) is not int or x < 0 for x in offsets)):
                raise ValueError(f'Invalid tensor metadata: {key}')
            begin, end = offsets
            nbytes = math.prod(shape) * _SIZES[name]
            if end < begin or end - begin != nbytes or 8 + length + end > size:
                raise ValueError(f'Invalid tensor offsets/size: {key}')
            entries[key] = TensorInfo(tuple(shape), _TYPES[name], nbytes, 8 + length + begin)
            ranges.append((begin, end))
        cursor = 0
        for begin, end in sorted(ranges):
            if begin != cursor:
                raise ValueError(f'Overlapping or non-contiguous data: {shard}')
            cursor = end
        if cursor != size - 8 - length or set(entries) != self._expected[shard]:
            raise ValueError(f'Shard/index tensor keys or payload mismatch: {shard}')
        while self._headers and self._header_bytes + length > self.max_header_bytes:
            _, (_, old_size) = self._headers.popitem(last=False)
            self._header_bytes -= old_size
        self._headers[shard] = entries, length
        self._header_bytes += length
        self._stats['header_reads'] += 1
        return entries

    def info(self, key):
        with self._lock:
            return self._header(self.weight_map[key])[key]

    def shape(self, key):
        return list(self.info(key).shape)

    def dtype(self, key):
        return self.info(key).dtype

    def _evict_one(self):
        shard, (context, _) = next(iter(self._readers.items()))
        context.__exit__(None, None, None)
        del self._readers[shard]
        if self.backend != 'file_readinto':
            self._stats['mapped_file_bytes'] -= self._identity[shard][0]

    def _reader(self, shard):
        self._check()
        path = self._path(shard)
        if shard in self._readers:
            self._stats['handle_hits'] += 1
            self._readers.move_to_end(shard)
            return self._readers[shard][1]
        size = 0 if self.backend == 'file_readinto' else self._identity[shard][0]
        if size > self.max_mapped_bytes:
            raise MemoryError('One shard exceeds mapped-file byte budget')
        while (self._readers and (len(self._readers) >= self.max_open_shards or
               self._stats['mapped_file_bytes'] + size > self.max_mapped_bytes)):
            self._evict_one()
        if self.backend == 'readonly_mmap':
            context = _ReadOnlyMapping(path, self._identity[shard])
        elif self.backend == 'file_readinto':
            context = path.open('rb', buffering=0)
            stat = os.fstat(context.fileno())
            if (stat.st_size, stat.st_mtime_ns, stat.st_ino) != self._identity[shard]:
                context.close()
                raise RuntimeError('Shard changed before opening the file')
        else:
            from safetensors import safe_open
            context = safe_open(path, framework='pt', device='cpu')
        reader = context.__enter__()
        self._readers[shard] = context, reader
        self._stats['shard_opens'] += 1
        self._stats['mapped_file_bytes'] += size
        self._stats['peak_mapped_file_bytes'] = max(self._stats['peak_mapped_file_bytes'],
                                                    self._stats['mapped_file_bytes'])
        return reader

    def read_into(self, key, destination, byte_offset=0):
        """Copy exact bytes synchronously into caller-owned CPU uint8 storage.

        Arbitrary byte offsets/lengths are valid; out-of-bounds, invalid destination
        or short source raise, never truncate. A copy error may partially modify
        destination: discard/re-fill it before use. No source aliases escape, even
        through exception tracebacks. Caller controls destination/pinned lifetime.
        """
        with self._lock:
            self._check()
            _positive(byte_offset, 'byte_offset', True)
            if (not isinstance(destination, torch.Tensor) or destination.device.type != 'cpu'
                    or destination.dtype != torch.uint8 or destination.layout != torch.strided
                    or not destination.is_contiguous() or destination.is_neg() or destination.is_conj()):
                raise ValueError('destination must be a contiguous writable CPU uint8 tensor')
            info = self.info(key)
            count = destination.numel()
            if byte_offset + count > info.nbytes:
                raise ValueError('read_into range exceeds tensor byte bounds')
            self._stats['read_into_calls'] += 1
            if not count:
                return 0
            try:
                if self.backend == 'readonly_mmap':
                    mapping = self._reader(self.weight_map[key])
                    # Check actual map size too, not only cached header metadata.
                    begin = info.offset + byte_offset
                    if begin + count > len(mapping):
                        raise OSError('Short readonly mapping')
                    root = memoryview(mapping)
                    source = target = None
                    try:
                        source = root[begin:begin + count]
                        target = memoryview(destination.numpy()).cast('B')
                        _copy_bytes(target, source)
                    finally:
                        if target is not None:
                            target.release()
                        if source is not None:
                            source.release()
                        root.release()
                    self._stats['mapped_copy_bytes_total'] += count
                elif self.backend == 'file_readinto':
                    reader = self._reader(self.weight_map[key])
                    reader.seek(info.offset + byte_offset)
                    with memoryview(destination.numpy()).cast('B') as target:
                        filled = 0
                        while filled < count:
                            with target[filled:] as rest:
                                got = reader.readinto(rest)
                            if not got:
                                raise OSError('Short safetensors file read')
                            filled += got
                else:
                    source = self._reader(self.weight_map[key]).get_tensor(key).reshape(-1).view(torch.uint8)
                    if byte_offset + count > source.numel():
                        raise OSError('Short safetensors tensor')
                    destination.reshape(-1).copy_(source[byte_offset:byte_offset + count])
            except BaseException:
                self._stats['read_into_failures'] += 1
                raise
            self._stats['read_into_bytes'] += count
            self._stats['logical_tensor_bytes'] += count
            return count

    def tensor(self, key):
        with self._lock:
            info = self.info(key)
            self._stats['tensor_requests'] += 1
            if key in self._cache:
                self._cache.move_to_end(key)
                self._stats['cache_hits'] += 1
                self._stats['logical_tensor_bytes'] += info.nbytes
                if self.backend != 'safe_open':
                    self._stats['owned_tensor_bytes_total'] += info.nbytes
                    return self._cache[key].clone()
                return self._cache[key]
            if self.backend != 'safe_open':
                tensor = torch.empty(info.shape, dtype=info.dtype, device='cpu')
                self.read_into(key, tensor.reshape(-1).view(torch.uint8))
                self._stats['owned_tensor_bytes_total'] += info.nbytes
            else:
                self._stats['logical_tensor_bytes'] += info.nbytes
                tensor = self._reader(self.weight_map[key]).get_tensor(key)
            if 0 < info.nbytes <= self.max_cache_bytes:
                while self._cache and self._stats['cache_bytes'] + info.nbytes > self.max_cache_bytes:
                    _, old = self._cache.popitem(last=False)
                    self._stats['cache_bytes'] -= old.numel() * old.element_size()
                if self.backend != 'safe_open':
                    self._cache[key] = tensor.clone()
                else:
                    tensor = tensor.clone()
                    self._cache[key] = tensor
                self._stats['cache_bytes'] += info.nbytes
            return tensor

    def stage(self, keys, *, pin_memory=False):
        """Nonblocking byte reservation; MemoryError means drain/split, not bypass.

        Copies are synchronous CPU copies; M2 may call in a bounded worker.
        Optional pinned allocations count in the same staging budget. They can
        initialize CUDA and are opt-in, never used by CPU-only tests.
        """
        with self._lock:
            self._check()
            keys = list(keys)
            if len(keys) != len(set(keys)):
                raise ValueError('Duplicate staging key')
            nbytes = sum(self.info(k).nbytes for k in keys)
            if self._stats['staging_bytes'] + nbytes > self.max_staging_bytes:
                raise MemoryError('CPU staging byte budget exceeded; drain leases or split group')
            self._stats['staging_bytes'] += nbytes
            self._stats['active_batches'] += 1
            self._stats['peak_staging_bytes'] = max(self._stats['peak_staging_bytes'], self._stats['staging_bytes'])
            tensors, start = {}, time.perf_counter()
            try:
                for key in keys:
                    info = self.info(key)
                    target = torch.empty(info.shape, dtype=info.dtype, device='cpu', pin_memory=pin_memory)
                    if self.backend == 'readonly_mmap':
                        self.read_into(key, target.reshape(-1).view(torch.uint8))
                    else:
                        target.copy_(self.tensor(key))
                    tensors[key] = target
            except BaseException:
                tensors.clear()
                self._stats['staging_bytes'] -= nbytes
                self._stats['active_batches'] -= 1
                raise
            self._stats['staged_bytes_total'] += nbytes
            self._stats['staging_seconds'] += time.perf_counter() - start
            return TensorBatch(self, tensors, nbytes)

    def iter_chunks(self, key, *, max_bytes):
        """CPU axis0 chunks: safe_open views; readonly_mmap owned, never pinned.

        A single row/scalar larger than max_bytes is rejected, never oversized.
        TensorBatch stage() is for whole tensors; this iterator is for the caller
        to split oversized tensors before copying through its own bounded slots.
        """
        _positive(max_bytes, 'max_bytes')
        info = self.info(key)
        if not info.shape:
            if info.nbytes > max_bytes:
                raise MemoryError('Scalar exceeds chunk budget')
            yield self.tensor(key)
            return
        # No rows means no slice, allocation or read, irrespective of the
        # hypothetical size of one row. Keep positive max_bytes validation above.
        if info.shape[0] == 0:
            return
        row_bytes = math.prod(info.shape[1:]) * torch.empty((), dtype=info.dtype, device='cpu').element_size()
        if row_bytes > max_bytes:
            raise MemoryError('One tensor row exceeds chunk budget')
        rows = max(1, max_bytes // max(1, row_bytes))
        for begin in range(0, info.shape[0], rows):
            with self._lock:
                self._check()
                if self.backend == 'readonly_mmap':
                    shape = (min(rows, info.shape[0]-begin), *info.shape[1:])
                    piece = torch.empty(shape, dtype=info.dtype, device='cpu')
                    self.read_into(key, piece.reshape(-1).view(torch.uint8), byte_offset=begin*row_bytes)
                    self._stats['owned_tensor_bytes_total'] += piece.numel() * piece.element_size()
                else:
                    piece = self._reader(self.weight_map[key]).get_slice(key)[begin:begin + rows]
                    self._stats['logical_tensor_bytes'] += piece.numel() * piece.element_size()
            yield piece

    def validate_keys(self, model):
        # state_dict holds only references (often meta); never materialize weights.
        state = model.state_dict()
        missing, unexpected = set(state) - set(self.weight_map), set(self.weight_map) - set(state)
        if missing or unexpected:
            raise ValueError(f'Checkpoint architecture mismatch: missing={sorted(missing)[:12]}, unexpected={sorted(unexpected)[:12]}')
        for key, tensor in state.items():
            if tuple(self.shape(key)) != tuple(tensor.shape):
                raise ValueError(f'Checkpoint shape mismatch for {key}')

    def stats_snapshot(self):
        with self._lock:
            return dict(self._stats, backend=self.backend, open_shards=len(self._readers),
                        max_open_shards=self.max_open_shards, max_mapped_bytes=self.max_mapped_bytes,
                        header_cache_bytes=self._header_bytes, cached_headers=len(self._headers),
                        closed=self.closed,
                        mapped_bytes_basis='live store-managed file extents, NOT resident/commit; OS page cache not bounded here',
                        owned_bytes_basis='cache/staging live; owned_tensor_bytes_total cumulative caller copies, not live memory',
                        io_bytes_basis='logical requested bytes; NOT physical disk reads')

    def close(self):
        with self._lock:
            if self._stats['active_batches']:
                raise RuntimeError('Release staging batches after H2D completion before closing store')
            while self._readers:
                self._evict_one()
            self._headers.clear()
            self._cache.clear()
            self._header_bytes = self._stats['cache_bytes'] = 0
            self.closed = True

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, *args):
        self.close()
