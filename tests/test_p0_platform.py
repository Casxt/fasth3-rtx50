"""CPU-only regression tests; never initialize CUDA during pytest collection."""
from types import SimpleNamespace
import sys
import pytest
from fasth3_platform import (GIB, MIB, configure_gpu_budget, check_gpu_headroom,
                             platform_info)


def fake_cuda(monkeypatch, free=12*GIB, total=16*GIB):
    calls = []
    cuda = SimpleNamespace(is_available=lambda: True,
        mem_get_info=lambda _: (free, total),
        set_per_process_memory_fraction=lambda fraction, device: calls.append((fraction, device)),
        memory_allocated=lambda _: 0, memory_reserved=lambda _: 0,
        max_memory_allocated=lambda _: 0)
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=cuda))
    monkeypatch.setattr('fasth3_platform.check_host_memory', lambda: {})
    return calls


def test_budget_caps_ten_gib(monkeypatch):
    calls = fake_cuda(monkeypatch)
    result = configure_gpu_budget()
    assert result.budget_bytes == 10*GIB
    assert result.reserve_bytes == 256*MIB
    assert calls == [(10/16, 0)]


def test_budget_respects_free_memory(monkeypatch):
    fake_cuda(monkeypatch, free=2*GIB)
    assert configure_gpu_budget().budget_bytes == 2*GIB - 256*MIB


@pytest.mark.parametrize('kwargs', [dict(max_gib=11), dict(max_gib=0),
    dict(max_gib=float('nan')), dict(max_gib=float('inf')), dict(reserve_mib=255)])
def test_reject_unsafe_budget(monkeypatch, kwargs):
    fake_cuda(monkeypatch)
    with pytest.raises(ValueError):
        configure_gpu_budget(**kwargs)


def test_reject_no_headroom(monkeypatch):
    fake_cuda(monkeypatch, free=128*MIB)
    with pytest.raises(MemoryError):
        configure_gpu_budget()


def test_runtime_headroom(monkeypatch):
    fake_cuda(monkeypatch)
    budget = configure_gpu_budget()
    fake_cuda(monkeypatch, free=128*MIB)
    with pytest.raises(MemoryError):
        check_gpu_headroom(budget)


def test_cpu_guard_no_cuda(monkeypatch):
    from fasth3_platform import BudgetGuard
    fake_cuda(monkeypatch)
    monkeypatch.setattr('fasth3_platform.check_host_memory', lambda: {'available_bytes': 4*GIB})
    with BudgetGuard(device='cpu') as guard:
        guard('encoder')
    assert guard.budget is None
    assert [s['phase'] for s in guard.samples] == ['startup', 'encoder', 'complete']


def test_reject_low_windows_commit_headroom(monkeypatch):
    from fasth3_platform import check_host_memory
    monkeypatch.setattr('fasth3_platform.platform.system', lambda: 'Windows')
    monkeypatch.setattr('fasth3_platform.windows_commit_info', lambda: {
        'commit_bytes': 9*GIB, 'commit_limit_bytes': 10*GIB,
        'commit_headroom_bytes': GIB, 'commit_source': 'mock'})
    with pytest.raises(MemoryError, match='commit headroom'):
        check_host_memory()


def test_reject_private_memory_above_hard_cap(monkeypatch):
    from fasth3_platform import BudgetGuard
    import psutil
    monkeypatch.setattr('fasth3_platform.check_host_memory', lambda: {})
    proc = SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=GIB, private=81*GIB),
                           children=lambda recursive: [])
    monkeypatch.setattr(psutil, 'Process', lambda: proc)
    with pytest.raises(MemoryError, match='private'):
        BudgetGuard(device='cpu')('private pressure')


def test_device_only_guard_checks_reserve_and_records_device_sample(monkeypatch):
    from fasth3_platform import BudgetGuard
    guard = BudgetGuard()
    with pytest.raises(RuntimeError, match='active CUDA budget'):
        guard.check_device('tensor')
    guard.budget = object()

    def check(budget):
        assert budget is guard.budget
        return {'free_bytes': GIB}

    monkeypatch.setattr('fasth3_platform.check_gpu_headroom', check)
    guard.check_device('tensor')
    assert guard.samples == [{'phase': 'tensor', 'gpu': {'free_bytes': GIB}}]

    def low_reserve(budget):
        raise MemoryError('reserve breached')

    monkeypatch.setattr('fasth3_platform.check_gpu_headroom', low_reserve)
    with pytest.raises(MemoryError, match='reserve'):
        guard.check_device('next tensor')


def test_environment_no_gpu_required():
    assert platform_info()['python_version'].startswith('3.')
