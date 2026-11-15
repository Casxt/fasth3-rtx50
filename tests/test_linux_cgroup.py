"""Container resource checks must use effective limits rather than host RAM."""
from types import SimpleNamespace

import pytest

import fasth3_platform as platform

GIB = platform.GIB


def write_group(directory, *, limit, current, inactive=0, version=2):
    directory.mkdir(parents=True, exist_ok=True)
    for name, value in ((('memory.max' if version == 2 else 'memory.limit_in_bytes'), limit),
                        (('memory.current' if version == 2 else 'memory.usage_in_bytes'), current)):
        (directory/name).write_text(str(value))
    key = 'inactive_file' if version == 2 else 'total_inactive_file'
    (directory/'memory.stat').write_text(f'{key} {inactive}\n')


def test_v2_credits_reclaimable_file_cache(tmp_path):
    root = tmp_path/'cgroup'
    write_group(root, limit=92*GIB, current=91*GIB, inactive=70*GIB)
    member = tmp_path/'membership'
    member.write_text('0::/\n')
    result = platform.linux_cgroup_memory(root, member)
    assert result['cgroup_effective_limit_bytes'] == 92*GIB
    assert result['cgroup_working_set_bytes'] == 21*GIB
    assert result['cgroup_available_bytes'] == 71*GIB


def test_parent_pressure_can_constrain_child(tmp_path):
    root = tmp_path/'cgroup'
    write_group(root, limit=100*GIB, current=99*GIB)
    write_group(root/'job', limit=80*GIB, current=20*GIB)
    member = tmp_path/'membership'
    member.write_text('0::/job\n')
    result = platform.linux_cgroup_memory(root, member)
    assert result['cgroup_available_bytes'] == GIB
    assert result['cgroup_effective_limit_bytes'] == 80*GIB
    assert result['cgroup_path'] == str(root.resolve())


def test_v2_host_root_without_memory_max_still_finds_job_limit(tmp_path):
    root = tmp_path/'cgroup'
    write_group(root/'job', limit=4*GIB, current=GIB)
    (root/'cgroup.controllers').write_text('memory cpu\n')
    member = tmp_path/'membership'
    member.write_text('0::/job\n')
    assert platform.linux_cgroup_memory(root, member)['cgroup_available_bytes'] == 3*GIB


def test_v1_reads_hierarchical_inactive_cache(tmp_path):
    root = tmp_path/'cgroup'
    write_group(root/'memory', limit=80*GIB, current=65*GIB, inactive=10*GIB, version=1)
    member = tmp_path/'membership'
    member.write_text('4:memory:/\n')
    result = platform.linux_cgroup_memory(root, member)
    assert result['cgroup_version'] == 1
    assert result['cgroup_available_bytes'] == 25*GIB


@pytest.mark.parametrize('limit', ['max', 9223372036854771712, 'bad', -1])
def test_unlimited_or_malformed_limits_fall_back(tmp_path, limit):
    root = tmp_path/'cgroup'
    write_group(root, limit=limit, current=GIB)
    member = tmp_path/'membership'
    member.write_text('0::/\n')
    assert platform.linux_cgroup_memory(root, member) == {}


def test_guard_obeys_container_pressure_even_on_large_host(monkeypatch):
    import psutil

    monkeypatch.setattr(platform.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(psutil, 'virtual_memory', lambda: SimpleNamespace(available=473*GIB, total=754*GIB))
    monkeypatch.setattr(platform, 'linux_cgroup_memory', lambda: {'cgroup_available_bytes': GIB})
    with pytest.raises(MemoryError, match='host/container memory'):
        platform.check_host_memory()


def test_guard_reports_host_and_container_scopes_separately(monkeypatch):
    import psutil

    monkeypatch.setattr(platform.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(psutil, 'virtual_memory', lambda: SimpleNamespace(available=473*GIB, total=754*GIB))
    monkeypatch.setattr(platform, 'linux_cgroup_memory', lambda: {
        'cgroup_available_bytes': 70*GIB, 'cgroup_effective_limit_bytes': 92*GIB})
    result = platform.check_host_memory()
    assert result['available_bytes'] == 70*GIB
    assert result['host_available_bytes'] == 473*GIB
    assert result['host_total_bytes'] == 754*GIB
    assert result['available_scope'] == 'cgroup'


@pytest.mark.parametrize(('quota', 'affinity', 'expected'), [('1600000 100000', 208, 16),
                                                          ('250000 100000', 32, 2),
                                                          ('max 100000', 6, 6)])
def test_cpu_quota_and_affinity(monkeypatch, tmp_path, quota, affinity, expected):
    root = tmp_path/'cgroup'
    root.mkdir()
    (root/'cpu.max').write_text(quota)
    member = tmp_path/'membership'
    member.write_text('0::/\n')
    monkeypatch.setattr(platform.os, 'cpu_count', lambda: 208)
    monkeypatch.setattr(platform.os, 'sched_getaffinity', lambda _: set(range(affinity)), raising=False)
    assert platform.effective_cpu_count(root, member) == expected


def test_linux_thread_defaults_preserve_explicit_configuration(monkeypatch):
    monkeypatch.setattr(platform.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(platform, 'effective_cpu_count', lambda: 4)
    monkeypatch.setenv('OMP_NUM_THREADS', '2')
    monkeypatch.delenv('MKL_NUM_THREADS', raising=False)
    platform.configure_cpu_threads()
    assert platform.os.environ['OMP_NUM_THREADS'] == '2'
    assert platform.os.environ['MKL_NUM_THREADS'] == '4'
