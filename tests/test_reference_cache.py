import torch

from fasth3_pipeline.reference_cache import ReferenceCache


def test_reference_cache_does_not_share_mutable_tensors_and_evicts_by_bytes():
    cache = ReferenceCache(16)
    first = torch.tensor([1., 2.])
    cache.put('first', ([first], []))
    first.zero_()
    got = cache.get('first')
    torch.testing.assert_close(got[0][0], torch.tensor([1., 2.]))
    got[0][0].zero_()
    cache.put('second', (torch.tensor([3., 4.]),))
    torch.testing.assert_close(cache.get('first')[0][0], torch.tensor([1., 2.]))
    cache.put('third', (torch.tensor([5., 6.]),))
    assert cache.get('second') is None
    assert cache.snapshot() == dict(bytes=16, hits=2, misses=1, evictions=1, limit_bytes=16)
