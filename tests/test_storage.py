"""Only synthetic CPU files. Lead may need full access for pytest tmp_path ACLs."""
import json
from pathlib import Path
import struct
from concurrent.futures import ThreadPoolExecutor
import pytest
import torch
from safetensors.torch import save_file

from fasth3_pipeline.storage import INDEX, PersistentTensorStore


def checkpoint(path, tensors=None):
    tensors = tensors or {'a': torch.arange(8, dtype=torch.float32),
                          'b': torch.arange(8, dtype=torch.float32)+10}
    mapping = {}
    for i, (key, value) in enumerate(tensors.items()):
        shard = f'part-{i}.safetensors'
        save_file({key: value}, path/shard)
        mapping[key] = shard
    (path/INDEX).write_text(json.dumps({'weight_map': mapping}), encoding='utf-8')
    return tensors


def raw_shard(path, header, payload=b''):
    data = json.dumps(header).encode()
    # Safetensors mandates 8-byte header alignment.
    data += b' ' * (-len(data) % 8)
    path.write_bytes(struct.pack('<Q', len(data))+data+payload)


def test_metadata_does_not_materialize_tensors_and_handles_reused(tmp_path):
    expected = checkpoint(tmp_path)
    with PersistentTensorStore(tmp_path) as store:
        assert store.shape('a') == [8]
        assert store.dtype('a') == torch.float32
        assert store.stats_snapshot()['shard_opens'] == 0
        assert store.stats_snapshot()['header_reads'] == 1
        for _ in range(3):
            torch.testing.assert_close(store.tensor('a'), expected['a'])
        stats = store.stats_snapshot()
        assert stats['shard_opens'] == 1
        assert stats['handle_hits'] == 2
        assert stats['header_reads'] == 1
        assert stats['cache_bytes'] == 0
        assert stats['logical_tensor_bytes'] == 3*32
        assert 'NOT physical' in stats['io_bytes_basis']


def test_tensor_outlives_eviction_and_explicit_close(tmp_path):
    expected = checkpoint(tmp_path)
    store = PersistentTensorStore(tmp_path, max_open_shards=1)
    first = store.tensor('a')
    second = store.tensor('b')
    assert store.stats_snapshot()['open_shards'] == 1
    store.close()
    store.close()
    torch.testing.assert_close(first, expected['a'])
    torch.testing.assert_close(second, expected['b'])
    assert store.stats_snapshot()['open_shards'] == 0
    with pytest.raises(RuntimeError, match='closed'):
        store.tensor('a')


def test_header_cache_eviction_is_bounded(tmp_path):
    checkpoint(tmp_path)
    lengths = [struct.unpack('<Q', (tmp_path/f'part-{i}.safetensors').read_bytes()[:8])[0]
               for i in range(2)]
    bound = max(lengths)
    with PersistentTensorStore(tmp_path, max_header_bytes=bound) as store:
        store.shape('a')
        store.shape('b')
        assert store.stats_snapshot()['header_cache_bytes'] <= bound
        store.shape('a')
        assert store.stats_snapshot()['header_reads'] == 3
        assert store.stats_snapshot()['shard_opens'] == 0


def test_owned_cache_lru_is_byte_bounded(tmp_path):
    checkpoint(tmp_path)
    with PersistentTensorStore(tmp_path, max_cache_bytes=32) as store:
        first = store.tensor('a')
        assert first.data_ptr() == store.tensor('a').data_ptr()
        assert store.stats_snapshot()['cache_hits'] == 1
        store.tensor('b')
        stats = store.stats_snapshot()
        assert stats['cache_bytes'] == 32
        store.tensor('a')
        assert store.stats_snapshot()['cache_bytes'] == 32
    assert store.stats_snapshot()['cache_bytes'] == 0


def test_staging_global_byte_limit_and_close_lifetime(tmp_path):
    expected = checkpoint(tmp_path)
    store = PersistentTensorStore(tmp_path, max_staging_bytes=32)
    batch = store.stage(['a'])
    torch.testing.assert_close(batch['a'], expected['a'])
    assert batch['a'].data_ptr() != store.tensor('a').data_ptr()
    assert not batch['a'].is_pinned()
    with pytest.raises(MemoryError, match='budget'):
        store.stage(['b'])
    with pytest.raises(RuntimeError, match='batches'):
        store.close()
    batch.close()
    batch.close()
    with pytest.raises(RuntimeError, match='closed'):
        batch['a']
    with store.stage(['b']) as second:
        assert second.nbytes == 32
    assert store.stats_snapshot()['staging_bytes'] == 0
    assert store.stats_snapshot()['peak_staging_bytes'] == 32
    store.close()


def test_staging_error_rolls_back_reservation(tmp_path, monkeypatch):
    checkpoint(tmp_path)
    with PersistentTensorStore(tmp_path, max_staging_bytes=64) as store:
        original = store.tensor
        def fail(key):
            if key == 'b':
                raise RuntimeError('injected read failure')
            return original(key)
        monkeypatch.setattr(store, 'tensor', fail)
        with pytest.raises(RuntimeError, match='injected'):
            store.stage(['a', 'b'])
        assert store.stats_snapshot()['active_batches'] == 0
        assert store.stats_snapshot()['staging_bytes'] == 0
        with pytest.raises(ValueError, match='Duplicate'):
            store.stage(['a', 'a'])


def test_chunked_large_tensor_and_tail(tmp_path):
    value = torch.arange(30, dtype=torch.float32).reshape(10, 3)
    checkpoint(tmp_path, {'a': value})
    with PersistentTensorStore(tmp_path) as store:
        chunks = list(store.iter_chunks('a', max_bytes=48))
        assert [tuple(x.shape) for x in chunks] == [(4, 3), (4, 3), (2, 3)]
        assert all(x.numel()*x.element_size() <= 48 for x in chunks)
        torch.testing.assert_close(torch.cat(chunks), value)
        with pytest.raises(MemoryError, match='row'):
            list(store.iter_chunks('a', max_bytes=11))


@pytest.mark.parametrize('value', [torch.tensor(2.), torch.empty(0, 3), torch.empty(3, 0)])
def test_scalar_and_empty_tensors(tmp_path, value):
    checkpoint(tmp_path, {'a': value})
    with PersistentTensorStore(tmp_path) as store:
        torch.testing.assert_close(store.tensor('a'), value)
        pieces = list(store.iter_chunks('a', max_bytes=8))
        if value.ndim == 0:
            torch.testing.assert_close(pieces[0], value)
        elif value.shape[0] == 0:
            assert not pieces
        else:
            torch.testing.assert_close(torch.cat(pieces), value)


@pytest.mark.parametrize('shape', [(0, 1 << 20), (0, 2, 1 << 20)])
def test_zero_rows_large_features_never_map_or_relax_budget(tmp_path, shape):
    # Zero storage bytes despite a hypothetical multi-MiB row. A tiny positive
    # budget is valid; invalid zero budgets and nonempty oversized rows are not.
    checkpoint(tmp_path, {'a': torch.empty(shape, dtype=torch.float32)})
    with PersistentTensorStore(tmp_path) as store:
        assert list(store.iter_chunks('a', max_bytes=1)) == []
        stats = store.stats_snapshot()
        assert stats['shard_opens'] == 0
        assert stats['logical_tensor_bytes'] == 0
        assert stats['staging_bytes'] == 0
        with pytest.raises(ValueError, match='max_bytes'):
            list(store.iter_chunks('a', max_bytes=0))


def test_validate_keys_and_shapes_without_model_load(tmp_path):
    model = torch.nn.Linear(4, 2, device='meta')
    checkpoint(tmp_path, {'weight': torch.zeros(2, 4), 'bias': torch.zeros(2)})
    with PersistentTensorStore(tmp_path) as store:
        store.validate_keys(model)
        assert store.stats_snapshot()['shard_opens'] == 0
        with pytest.raises(ValueError, match='shape'):
            store.validate_keys(torch.nn.Linear(3, 2, device='meta'))
        with pytest.raises(ValueError, match='architecture'):
            store.validate_keys(torch.nn.Linear(4, 2, bias=False, device='meta'))


def test_threadsafe_repeated_read_and_stats(tmp_path):
    checkpoint(tmp_path)
    with PersistentTensorStore(tmp_path, max_open_shards=1) as store:
        def read(i):
            key = 'a' if i % 2 == 0 else 'b'
            return store.tensor(key).clone().sum().item()
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(read, range(24)))
        assert results == [28 if i % 2 == 0 else 108 for i in range(24)]
        assert store.stats_snapshot()['open_shards'] <= 1
        assert store.stats_snapshot()['tensor_requests'] == 24


@pytest.mark.parametrize('shard', ['../evil.safetensors', 'sub/a.safetensors',
                                  'sub\\a.safetensors', 'C:\\a.safetensors',
                                  '/a.safetensors', 'a.txt', ''])
def test_reject_unsafe_paths(tmp_path, shard):
    (tmp_path/INDEX).write_text(json.dumps({'weight_map': {'a': shard}}))
    with pytest.raises(ValueError, match='[Uu]nsafe'):
        PersistentTensorStore(tmp_path)


def test_duplicate_index_keys_rejected(tmp_path):
    (tmp_path/INDEX).write_text('{"weight_map":{"a":"a.safetensors","a":"b.safetensors"}}')
    with pytest.raises(ValueError, match='Duplicate'):
        PersistentTensorStore(tmp_path)


def test_file_change_detected(tmp_path):
    checkpoint(tmp_path)
    with PersistentTensorStore(tmp_path) as store:
        store.shape('a')
        # No mapped handle exists yet; changing an open mapping is not supported.
        with (tmp_path/'part-0.safetensors').open('ab') as stream:
            stream.write(b'x')
        with pytest.raises(RuntimeError, match='changed'):
            store.tensor('a')


def test_index_shard_mismatch_rejected(tmp_path):
    checkpoint(tmp_path)
    index = json.loads((tmp_path/INDEX).read_text())
    index['weight_map']['a'] = index['weight_map']['b']
    (tmp_path/INDEX).write_text(json.dumps(index))
    with PersistentTensorStore(tmp_path) as store:
        with pytest.raises(ValueError, match='mismatch'):
            store.shape('a')


@pytest.mark.parametrize('item', [
    {'dtype': 'F32', 'shape': [-1], 'data_offsets': [0, 4]},
    {'dtype': 'F32', 'shape': [2], 'data_offsets': [0, 4]},
    {'dtype': 'F32', 'shape': [1], 'data_offsets': [4, 8]},
    {'dtype': 'UNKNOWN', 'shape': [1], 'data_offsets': [0, 4]},
    {'dtype': 'F32', 'shape': [True], 'data_offsets': [0, 4]},
])
def test_malformed_header_rejected_before_tensor(tmp_path, item):
    (tmp_path/INDEX).write_text(json.dumps({'weight_map': {'a': 'a.safetensors'}}))
    raw_shard(tmp_path/'a.safetensors', {'a': item}, b'\x00'*4)
    with PersistentTensorStore(tmp_path) as store:
        with pytest.raises(ValueError):
            store.shape('a')
        assert store.stats_snapshot()['shard_opens'] == 0


def test_oversized_header_and_index_limits(tmp_path):
    checkpoint(tmp_path)
    with pytest.raises(ValueError, match='index'):
        PersistentTensorStore(tmp_path, max_index_bytes=2)
    with PersistentTensorStore(tmp_path, max_header_bytes=2) as store:
        with pytest.raises(ValueError, match='header'):
            store.shape('a')


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16, torch.float16,
    torch.float8_e4m3fn, torch.float8_e5m2, torch.int64, torch.uint8, torch.bool])
def test_readonly_owned_tensor_dtype_and_exact_bytes(tmp_path, dtype):
    value = torch.arange(8, dtype=torch.float32).reshape(2, 4).to(dtype)
    checkpoint(tmp_path, {'a': value})
    with PersistentTensorStore(tmp_path, backend='readonly_mmap') as store:
        copied = store.tensor('a')
        assert copied.dtype == dtype and copied.shape == value.shape
        assert torch.equal(copied.reshape(-1).view(torch.uint8), value.reshape(-1).view(torch.uint8))
        stats = store.stats_snapshot()
        assert stats['mapped_copy_bytes_total'] == value.numel()*value.element_size()
        assert stats['owned_tensor_bytes_total'] == stats['mapped_copy_bytes_total']
        assert stats['cache_bytes'] == stats['staging_bytes'] == 0
    assert store.stats_snapshot()['mapped_file_bytes'] == 0
    assert copied.device.type == 'cpu'  # Still valid after mapping has closed.


@pytest.mark.parametrize('backend', ['safe_open', 'readonly_mmap', 'file_readinto'])
def test_read_into_exact_partial_bytes_and_destination_slice(tmp_path, backend):
    values = checkpoint(tmp_path)
    expected = values['a'].view(torch.uint8)
    storage = torch.full((16,), 123, dtype=torch.uint8)
    with PersistentTensorStore(tmp_path, backend=backend) as store:
        count = store.read_into('a', storage[2:10], byte_offset=3)
        assert type(count) is int and count == 8
        assert torch.equal(storage[2:10], expected[3:11])
        assert (storage[:2] == 123).all() and (storage[10:] == 123).all()
        multidimensional = torch.empty(2, 4, dtype=torch.uint8)
        assert store.read_into('a', multidimensional, byte_offset=24) == 8
        assert torch.equal(multidimensional.reshape(-1), expected[24:32])
        assert store.stats_snapshot()['read_into_bytes'] == 16
        assert store.stats_snapshot()['owned_tensor_bytes_total'] == 0


@pytest.mark.parametrize('bad_destination', [None, torch.empty(2),
    torch.empty(2, dtype=torch.uint8, device='meta'),
    torch.empty(2, 3, dtype=torch.uint8).t(), torch.zeros(2, dtype=torch.uint8).to_sparse()])
def test_read_into_rejects_invalid_destinations(tmp_path, bad_destination):
    checkpoint(tmp_path)
    with PersistentTensorStore(tmp_path, backend='readonly_mmap') as store:
        with pytest.raises(ValueError, match='destination'):
            store.read_into('a', bad_destination)
        assert store.stats_snapshot()['shard_opens'] == 0


@pytest.mark.parametrize('offset', [-1, True, 0.5, 33])
def test_read_into_rejects_invalid_offsets_without_mapping(tmp_path, offset):
    checkpoint(tmp_path)
    with PersistentTensorStore(tmp_path, backend='readonly_mmap') as store:
        with pytest.raises(ValueError):
            store.read_into('a', torch.empty(1, dtype=torch.uint8), byte_offset=offset)
        assert store.stats_snapshot()['shard_opens'] == 0


def test_read_into_bounds_closed_unknown_and_empty(tmp_path):
    checkpoint(tmp_path, {'a': torch.empty(0, 1 << 20)})
    store = PersistentTensorStore(tmp_path, backend='readonly_mmap')
    assert store.read_into('a', torch.empty(0, dtype=torch.uint8)) == 0
    assert store.stats_snapshot()['shard_opens'] == 0
    with pytest.raises(ValueError, match='bounds'):
        store.read_into('a', torch.empty(1, dtype=torch.uint8))
    with pytest.raises(KeyError):
        store.read_into('missing', torch.empty(0, dtype=torch.uint8))
    store.close()
    with pytest.raises(RuntimeError, match='closed'):
        store.read_into('a', torch.empty(0, dtype=torch.uint8))


def test_read_into_short_source_is_error_not_truncation(tmp_path, monkeypatch):
    checkpoint(tmp_path)
    with PersistentTensorStore(tmp_path, backend='readonly_mmap') as store:
        short_length = store.info('a').offset + 3
        monkeypatch.setattr(store, '_reader', lambda shard: bytes(short_length))
        destination = torch.full((4,), 111, dtype=torch.uint8)
        with pytest.raises(OSError, match='Short'):
            store.read_into('a', destination)
        assert (destination == 111).all()
        assert store.stats_snapshot()['read_into_bytes'] == 0
        assert store.stats_snapshot()['read_into_failures'] == 1


def test_readonly_mapping_source_cannot_be_written_and_no_alias_escapes(tmp_path):
    expected = checkpoint(tmp_path)
    source_path = tmp_path/'part-0.safetensors'
    before = source_path.read_bytes()
    with PersistentTensorStore(tmp_path, backend='readonly_mmap', max_cache_bytes=32) as store:
        first = store.tensor('a')
        mapping = next(iter(store._readers.values()))[1]
        with pytest.raises((TypeError, ValueError)):
            mapping[store.info('a').offset] = 0
        first.add_(100)
        second = store.tensor('a')
        torch.testing.assert_close(second, expected['a'])
        assert second.data_ptr() != first.data_ptr()
        second.zero_()
        torch.testing.assert_close(store.tensor('a'), expected['a'])
    assert mapping.closed
    assert source_path.read_bytes() == before


def test_readonly_eviction_and_mapped_byte_limit(tmp_path):
    expected = checkpoint(tmp_path)
    sizes = [(tmp_path/f'part-{i}.safetensors').stat().st_size for i in range(2)]
    store = PersistentTensorStore(tmp_path, backend='readonly_mmap', max_open_shards=2,
                                  max_mapped_bytes=max(sizes))
    first = store.tensor('a')
    old_context, old_map = next(iter(store._readers.values()))
    second = store.tensor('b')
    assert old_context.file.closed and old_map.closed
    stats = store.stats_snapshot()
    assert stats['open_shards'] == 1
    assert stats['mapped_file_bytes'] == sizes[1]
    assert stats['peak_mapped_file_bytes'] <= max(sizes)
    store.close()
    torch.testing.assert_close(first, expected['a'])
    torch.testing.assert_close(second, expected['b'])
    with PersistentTensorStore(tmp_path, backend='readonly_mmap', max_mapped_bytes=1) as too_small:
        with pytest.raises(MemoryError, match='mapped-file'):
            too_small.read_into('a', torch.empty(1, dtype=torch.uint8))
        assert too_small.stats_snapshot()['mapped_file_bytes'] == 0


def test_readonly_copy_error_releases_views_even_with_traceback_alive(tmp_path, monkeypatch):
    import fasth3_pipeline.storage as module
    checkpoint(tmp_path)
    path = tmp_path/'part-0.safetensors'
    before = path.read_bytes()
    retained = []
    def fail(destination, source):
        retained.extend((destination, source))
        destination[:1] = source[:1]
        raise OSError('injected partial memcpy error')
    store = PersistentTensorStore(tmp_path, backend='readonly_mmap')
    monkeypatch.setattr(module, '_copy_bytes', fail)
    with pytest.raises(OSError, match='partial') as error:
        store.read_into('a', torch.empty(8, dtype=torch.uint8))
    assert error.value.__traceback__ is not None
    stats = store.stats_snapshot()
    assert stats['read_into_failures'] == 1 and stats['read_into_bytes'] == 0
    context, mapping = next(iter(store._readers.values()))
    store.close()  # Must not raise BufferError from exception-held source views.
    assert context.file.closed and mapping.closed
    for view in retained:
        with pytest.raises(ValueError):
            view.tobytes()  # Released memoryviews, not dangling ndarray aliases.
    assert path.read_bytes() == before


def test_readonly_mapping_creation_failure_closes_fd(tmp_path, monkeypatch):
    import fasth3_pipeline.storage as module
    checkpoint(tmp_path)
    captured = []
    original_open = Path.open
    def track_open(path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        if path.suffix == '.safetensors':
            captured.append(stream)
        return stream
    def fail_map(*args, **kwargs):
        raise OSError('injected mmap failure')
    monkeypatch.setattr(Path, 'open', track_open)
    monkeypatch.setattr(module.mmap, 'mmap', fail_map)
    with PersistentTensorStore(tmp_path, backend='readonly_mmap') as store:
        with pytest.raises(OSError, match='mmap failure'):
            store.read_into('a', torch.empty(1, dtype=torch.uint8))
        assert store.stats_snapshot()['open_shards'] == 0
        assert store.stats_snapshot()['mapped_file_bytes'] == 0
    assert captured and all(stream.closed for stream in captured)


def test_readonly_chunks_stage_and_threaded_destinations(tmp_path):
    value = torch.arange(30, dtype=torch.float32).reshape(10, 3)
    checkpoint(tmp_path, {'a': value})
    with PersistentTensorStore(tmp_path, backend='readonly_mmap', max_staging_bytes=120) as store:
        chunks = list(store.iter_chunks('a', max_bytes=48))
        assert [x.shape[0] for x in chunks] == [4, 4, 2]
        torch.testing.assert_close(torch.cat(chunks), value)
        with store.stage(['a']) as batch:
            torch.testing.assert_close(batch['a'], value)
        def read(index):
            destination = torch.empty(4, dtype=torch.uint8)
            assert store.read_into('a', destination, byte_offset=index*4) == 4
            return destination.view(torch.float32).item()
        with ThreadPoolExecutor(max_workers=4) as pool:
            assert list(pool.map(read, range(30))) == list(range(30))
        assert store.stats_snapshot()['staging_bytes'] == 0


def test_invalid_backend_before_filesystem_access():
    with pytest.raises(ValueError, match='backend'):
        PersistentTensorStore('unused', backend='writable_mmap')


@pytest.mark.parametrize('option,value', [('max_open_shards', 0), ('max_cache_bytes', -1),
                                         ('max_staging_bytes', 0), ('max_header_bytes', True)])
def test_invalid_budgets_before_filesystem_access(option, value):
    with pytest.raises(ValueError):
        PersistentTensorStore('unused', **{option: value})
