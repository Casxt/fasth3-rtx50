"""Measure preallocated CPU/CUDA copies without loading model weights.

Run with the project's Python: python tools/benchmark_transfers.py --output out.json
CUDA events measure the stream interval; wall time includes submission and host
staging. In particular, a pageable copy's event interval is not pure DMA time.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fasth3_platform import (MIB, GIB, check_gpu_headroom, check_host_memory,
                             configure_cpu_threads, configure_gpu_budget,
                             effective_cpu_count, _normalize_gpu_uuid,
                             _cgroup_directories)


def command(args):
    try:
        return subprocess.run(args, capture_output=True, text=True,
                              timeout=10, check=True,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def gpu_selector(torch, device):
    rows = command(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"])
    wanted = _normalize_gpu_uuid(torch.cuda.get_device_properties(device).uuid)
    matches = [row[0].strip() for row in csv.reader((rows or "").splitlines())
               if len(row) == 2 and _normalize_gpu_uuid(row[1]) == wanted]
    if len(matches) != 1:
        raise RuntimeError("Cannot match CUDA device to NVIDIA telemetry")
    return matches[0]


def gpu_sample(selector):
    fields = ["name", "driver_version", "pci.bus_id", "memory.total", "memory.used",
              "utilization.gpu", "utilization.memory", "pcie.link.gen.current",
              "pcie.link.gen.max", "pcie.link.width.current", "pcie.link.width.max",
              "power.draw", "clocks.sm", "clocks.mem", "temperature.gpu"]
    text = command(["nvidia-smi", "-i", selector, "--query-gpu=" + ",".join(fields),
                    "--format=csv,noheader,nounits"])
    if not text:
        return {"available": False}
    values = next(csv.reader(text.splitlines()))
    result = dict(zip(fields, (v.strip() for v in values)))
    result["sampled_at_utc"] = datetime.now(timezone.utc).isoformat()
    return result


def hardware(torch, device, selector):
    if platform.system() == "Windows":
        cpu = command(["powershell", "-NoProfile", "-Command",
                       "(Get-CimInstance Win32_Processor).Name"])
    else:
        try:
            cpu = next(line.split(":", 1)[1].strip() for line in
                       Path("/proc/cpuinfo").read_text().splitlines()
                       if line.startswith("model name"))
        except (OSError, StopIteration):
            cpu = platform.processor()
    memory = check_host_memory()
    import psutil
    memory.setdefault("host_total_bytes", psutil.virtual_memory().total)
    # No machine identifiers, user paths, credentials or cgroup names in reports.
    memory = {k: v for k, v in memory.items() if k != "cgroup_path"}
    affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    if platform.system() == "Windows":
        try:
            import psutil
            affinity = len(psutil.Process().cpu_affinity())
        except (OSError, AttributeError):
            pass
    gpu = gpu_sample(selector)
    numa = {}
    quotas = []
    if platform.system() == "Linux" and gpu.get("pci.bus_id"):
        domain, bus, slot = gpu["pci.bus_id"].split(":")
        base = Path("/sys/bus/pci/devices") / f"{int(domain, 16):04x}:{bus}:{slot}"
        for name in ("numa_node", "local_cpulist"):
            try:
                numa[name] = (base / name).read_text().strip()
            except OSError:
                numa[name] = None
        topology = command(["nvidia-smi", "topo", "-m"])
        for line in (topology or "").splitlines():
            fields = line.split()
            if fields and fields[0] == f"GPU{selector}" and len(fields) >= 5:
                numa.update(topology_cpu_affinity=fields[-3],
                            topology_numa_affinity=fields[-2], topology_gpu_numa_id=fields[-1])
        version, directories = _cgroup_directories("/sys/fs/cgroup", "/proc/self/cgroup", "cpu")
        for directory in directories:
            try:
                if version == 2:
                    quota, period = (directory / "cpu.max").read_text().split()
                else:
                    quota = (directory / "cpu.cfs_quota_us").read_text().strip()
                    period = (directory / "cpu.cfs_period_us").read_text().strip()
                if int(quota) > 0 and int(period) > 0:
                    quotas.append({"quota_us": int(quota), "period_us": int(period)})
            except (OSError, ValueError):
                continue
    return {"os": platform.platform(), "python": platform.python_version(),
            "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "cpu_model": cpu, "cpu_host_visible_logical": os.cpu_count(),
            "cpu_affinity_count": affinity, "cpu_effective_count": effective_cpu_count(),
            "cpu_cgroup_quotas": quotas,
            "torch_threads": torch.get_num_threads(), "memory": memory,
            "gpu": gpu, "gpu_total_bytes": torch.cuda.get_device_properties(device).total_memory,
            "gpu_numa": numa}


def summary(times_ms, count):
    median = statistics.median(times_ms)
    seconds = median / 1000
    return {"median_ms": median, "min_ms": min(times_ms), "max_ms": max(times_ms),
            "mean_ms": statistics.mean(times_ms),
            "GiB_per_s": count / GIB / seconds, "GB_per_s": count / 1e9 / seconds}


def measure(torch, host, gpu, direction, count, repeats, warmup):
    cpu = host[:count]
    cuda = gpu[:count]
    expected = 37 if direction == "H2D" else 91
    cpu.fill_(expected if direction == "H2D" else 0)
    cuda.fill_(0 if direction == "H2D" else expected)
    source, target = (cpu, cuda) if direction == "H2D" else (cuda, cpu)
    for _ in range(warmup):
        target.copy_(source, non_blocking=True)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    # Materialize both event handles outside timed samples.
    start.record()
    end.record()
    end.synchronize()
    wall, events = [], []
    for _ in range(repeats):
        torch.cuda.synchronize()
        before = time.perf_counter_ns()
        start.record()
        target.copy_(source, non_blocking=True)
        end.record()
        end.synchronize()
        wall.append((time.perf_counter_ns() - before) / 1e6)
        events.append(start.elapsed_time(end))
    # Full payload verification and all allocations are outside measurements.
    valid = bool((target == expected).all().item())
    if not valid:
        raise RuntimeError("Copy verification failed")
    return {"direction": direction, "host_memory": "pinned" if host.is_pinned() else "pageable",
            "bytes": count, "MiB": count / MIB, "repeats": repeats,
            "cuda_event": summary(events, count), "cpu_wall": summary(wall, count),
            "verified": valid}


def link_under_load(torch, host, gpu, selector):
    """Sample PCIe state during a separate copy loop, outside benchmark timing."""
    samples, stop = [], threading.Event()

    def sample():
        for _ in range(3):
            time.sleep(.15)
            samples.append(gpu_sample(selector))
        stop.set()

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    started = time.perf_counter()
    while not stop.is_set() and time.perf_counter() - started < 15:
        gpu.copy_(host, non_blocking=True)
        host.copy_(gpu, non_blocking=True)
        torch.cuda.synchronize()
    thread.join(timeout=1)
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sizes-mib", type=int, nargs="+", default=[1, 16, 64, 256])
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--condition", default="No concurrent model benchmark; inspect GPU samples.")
    args = parser.parse_args()
    if min(args.sizes_mib) <= 0 or max(args.sizes_mib) > 512 or args.repeats < 3 or args.warmup < 1:
        parser.error("Use 1..512 MiB buffers, at least 3 repeats and 1 warmup")
    configure_cpu_threads()
    import torch

    torch.set_num_threads(min(8, effective_cpu_count()))
    torch.cuda.set_device(args.device)
    budget = configure_gpu_budget(args.device, max_gib=10, reserve_mib=256)
    # Match internally by UUID, but never serialize that machine identifier.
    selector = gpu_selector(torch, args.device)
    inventory = hardware(torch, args.device, selector)
    count = max(args.sizes_mib) * MIB
    if 2*count + MIB > budget.available_bytes():
        raise MemoryError("Insufficient GPU headroom for the copy and verification buffers")
    allocated_at = time.perf_counter()
    pageable = torch.empty(count, dtype=torch.uint8).fill_(37)
    pinned = torch.empty(count, dtype=torch.uint8, pin_memory=True).fill_(37)
    gpu = torch.empty(count, dtype=torch.uint8, device=args.device).fill_(91)
    torch.cuda.synchronize()
    setup_seconds = time.perf_counter() - allocated_at
    results = []
    for size in args.sizes_mib:
        for host in (pageable, pinned):
            for direction in ("H2D", "D2H"):
                results.append(measure(torch, host, gpu, direction, size*MIB,
                                       args.repeats, args.warmup))
        check_gpu_headroom(budget)
    load = link_under_load(torch, pinned, gpu, selector)
    result = {"schema_version": 1, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "run_condition": args.condition, "hardware": inventory,
              "measurement": {"warmup_per_case": args.warmup, "repeats_per_case": args.repeats,
                              "setup_seconds_excluded": setup_seconds,
                              "preallocated_gpu_bytes": count, "preallocated_host_bytes": 2*count,
                              "gpu_budget": budget.as_dict(), "non_blocking_requested": True,
                              "cuda_event_scope": "Stream interval, including possible staging/submission gaps.",
                              "cpu_wall_scope": "Event submission, copy call, and wait for completion.",
                              "telemetry_scope": "Before/after and separate transfer load; none inside timed samples."},
              "pcie_under_copy_load": load, "gpu_after": gpu_sample(selector),
              "memory_after": check_gpu_headroom(budget), "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"cases": len(results), "verified": all(r["verified"] for r in results),
                      "output_file": args.output.name}))


if __name__ == "__main__":
    main()
