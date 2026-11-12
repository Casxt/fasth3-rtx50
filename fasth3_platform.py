"""Native Windows/Linux capability checks and conservative tiny-probe budgets.

Importing this module does not import torch or initialize CUDA. Call
configure_gpu_budget() *before* allocating CUDA tensors. Limits apply to PyTorch's
allocator, not arbitrary CUDA libraries; keep non-PyTorch probes tiny as well.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
import os
import platform
import shutil

MIB = 1 << 20
GIB = 1 << 30


@dataclass(frozen=True)
class GPUBudget:
    device: int
    budget_bytes: int
    reserve_bytes: int
    free_bytes: int
    total_bytes: int

    def as_dict(self):
        return asdict(self)

    def available_bytes(self):
        import torch

        free, _ = torch.cuda.mem_get_info(self.device)
        return min(self.budget_bytes - torch.cuda.memory_allocated(self.device), free - self.reserve_bytes)


def windows_commit_info() -> dict:
    """System committed pages/limit from Win32 GetPerformanceInfo, not swap size."""
    import ctypes
    from ctypes import wintypes
    class PerformanceInformation(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD)] + [(name, ctypes.c_size_t) for name in
            ("CommitTotal", "CommitLimit", "CommitPeak", "PhysicalTotal",
             "PhysicalAvailable", "SystemCache", "KernelTotal", "KernelPaged",
             "KernelNonpaged", "PageSize")] + [(name, wintypes.DWORD) for name in
            ("HandleCount", "ProcessCount", "ThreadCount")]
    info = PerformanceInformation()
    info.cb = ctypes.sizeof(info)
    get_info = ctypes.WinDLL("psapi", use_last_error=True).GetPerformanceInfo
    get_info.argtypes = [ctypes.POINTER(PerformanceInformation), wintypes.DWORD]
    get_info.restype = wintypes.BOOL
    if not get_info(ctypes.byref(info), info.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return {"commit_bytes": info.CommitTotal * info.PageSize,
            "commit_limit_bytes": info.CommitLimit * info.PageSize,
            "commit_headroom_bytes": (info.CommitLimit-info.CommitTotal)*info.PageSize,
            "commit_source": "Windows psapi.GetPerformanceInfo (system-wide)"}


def check_host_memory(min_available_gib: float = 2.0) -> dict:
    """Refuse probes when host memory is under pressure; never allocate models."""
    import psutil
    available = psutil.virtual_memory().available
    if min_available_gib < 0 or not __import__('math').isfinite(min_available_gib):
        raise ValueError("min_available_gib must be finite and nonnegative")
    if available < min_available_gib * GIB:
        raise MemoryError(f"Host available memory {available / GIB:.2f} GiB below {min_available_gib} GiB")
    memory = psutil.Process().memory_info()
    result = {"available_bytes": available, "process_rss_bytes": memory.rss,
              "process_private_bytes": getattr(memory, "private", None)}
    if platform.system() == "Windows":
        result.update(windows_commit_info())
        if result["commit_headroom_bytes"] < 2 * GIB:
            raise MemoryError("Windows system commit headroom is below 2 GiB")
    else:
        result["commit_source"] = "unavailable on this platform; RSS/available guard only"
    return result


def configure_gpu_budget(device: int = 0, max_gib: float = 10.0,
                         reserve_mib: int = 256) -> GPUBudget:
    import math
    import torch
    if not math.isfinite(max_gib) or not 0 < max_gib <= 10:
        raise ValueError("max_gib must be finite, positive and <=10")
    if reserve_mib < 256:
        raise ValueError("reserve_mib must be >=256")
    check_host_memory()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; install the pinned Windows CUDA torch wheel")
    free, total = torch.cuda.mem_get_info(device)
    reserve = reserve_mib * MIB
    budget = min(int(max_gib * GIB), free - reserve)
    if budget <= 0:
        raise MemoryError("Insufficient free GPU memory after reserve")
    torch.cuda.set_per_process_memory_fraction(budget / total, device)
    return GPUBudget(device, budget, reserve, free, total)


def check_gpu_headroom(budget: GPUBudget) -> dict:
    import torch
    free, _ = torch.cuda.mem_get_info(budget.device)
    if free < budget.reserve_bytes:
        raise MemoryError("GPU free memory is below the configured safety reserve")
    return {"free_bytes": free,
            "allocated_bytes": torch.cuda.memory_allocated(budget.device),
            "reserved_bytes": torch.cuda.memory_reserved(budget.device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(budget.device)}


def _normalize_gpu_uuid(value):
    """Match CUDA UUID strings/objects/raw bytes to NVML's GPU-prefixed UUIDs."""
    import re

    raw = getattr(value, 'bytes', value)
    if isinstance(raw, (bytes, bytearray, memoryview)):
        if len(raw) == 16:
            return bytes(raw).hex()
        value = bytes(raw).decode('ascii')
    text = str(value).strip().strip('\0').lower()
    canonical = re.search(r'[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}|[0-9a-f]{32}', text)
    if canonical:
        return canonical.group().replace('-', '')
    return text.removeprefix('gpu-').replace('-', '')


def _query_gpu_board_memory(device: int, receipt=None):
    """One bounded NVML-backed sample, matched to the CUDA device UUID."""
    import csv
    import math
    import subprocess
    import torch

    try:
        get_properties = getattr(torch.cuda, 'get_device_properties', None)
        uuid = getattr(get_properties(device), 'uuid', None) if get_properties is not None else None
        selector = _normalize_gpu_uuid(uuid) if uuid is not None else None
        by_index = False
        visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        if selector is None and visible is not None:
            selector = visible.split(',')[device].strip()
            by_index = selector.isdigit()
            if not by_index:
                selector = _normalize_gpu_uuid(selector)
        if selector is None:
            selector = str(device)
            by_index = True
        if receipt is not None:
            receipt['board_query_selector'] = selector
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.total',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True, timeout=2,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        candidates = []
        for row in csv.reader(result.stdout.splitlines()):
            if len(row) != 4:
                continue
            index, board_uuid, used, total = (value.strip() for value in row)
            matches = (index == selector if by_index else
                       _normalize_gpu_uuid(board_uuid).startswith(selector))
            if matches:
                used, total = float(used) * MIB, float(total) * MIB
                if math.isfinite(used) and math.isfinite(total) and 0 <= used <= total:
                    candidates.append((int(used), int(total)))
        if len(candidates) == 1:
            return candidates[0]
        if receipt is not None:
            receipt['board_query_error'] = ('GPU selector did not match a valid memory row' if not candidates else
                                            'GPU selector matched multiple memory rows')
    except (OSError, subprocess.SubprocessError, AttributeError, RuntimeError,
            ValueError, IndexError, UnicodeError) as error:
        if receipt is not None:
            receipt['board_query_error'] = f'{type(error).__name__}: {error}'
    return None


@contextmanager
def dit_gpu_budget(guard, receipt: dict):
    """Temporarily tighten DiT's allocator cap using whole-board occupancy.

    The caller exits this context after releasing DiT buffers, including on an
    early diagnostic stop. CPU and lightweight mocked guards remain no-ops.
    """
    previous = getattr(guard, 'budget', None)
    if not isinstance(previous, GPUBudget):
        yield
        return
    import torch

    device = previous.device
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info(device)
    reserved = torch.cuda.memory_reserved(device)
    board = _query_gpu_board_memory(device, receipt)
    used, board_total = board if board is not None else (total - free, total)
    external = max(0, used - reserved)
    target, margin = 23 * GIB // 2, 128 * MIB
    reserve = max(previous.reserve_bytes, 256 * MIB)
    get_fraction = getattr(torch.cuda, 'get_per_process_memory_fraction', None)
    previous_fraction = (get_fraction(device) if get_fraction is not None else
                         previous.budget_bytes / total)
    cap = min(previous.budget_bytes, int(previous_fraction * total), 10 * GIB,
              min(target, board_total - reserve) - external - margin,
              reserved + free - reserve)
    receipt.update(source='nvidia-smi' if board is not None else 'cuda.mem_get_info',
                   board_used_bytes=used, external_bytes=external,
                   allocator_cap_bytes=cap, previous_allocator_cap_bytes=int(previous_fraction * total),
                   target_bytes=target, additional_margin_bytes=margin,
                   reserve_bytes=reserve, restored=False)
    if cap <= 0 or torch.cuda.memory_allocated(device) > cap:
        raise MemoryError('DiT cannot fit within the whole-board GPU memory target')
    try:
        torch.cuda.set_per_process_memory_fraction(cap / total, device)
        guard.budget = replace(previous, budget_bytes=cap, free_bytes=free,
                               total_bytes=total, reserve_bytes=reserve)
        yield
    finally:
        try:
            torch.cuda.set_per_process_memory_fraction(previous_fraction, device)
            receipt['restored'] = True
        finally:
            guard.budget = previous


class BudgetGuard:
    """Stage guard/context manager for bounded P1 model transfers.

    with BudgetGuard(device="cuda:0") as guard:
        guard("before encoder transfer")
        # load one bounded stage, then release it before the next stage

    CPU RSS is not system-wide/Windows file-cache accounting; both process-tree
    RSS and system available memory are checked, and metrics state this limit.
    """
    def __init__(self, device="cuda:0", max_gib=10.0, reserve_mib=256,
                 max_host_gib=80.0, preferred_host_gib=64.0):
        import math
        if not math.isfinite(max_host_gib) or not 0 < max_host_gib <= 80:
            raise ValueError("max_host_gib must be finite, positive and <=80")
        if not 0 < preferred_host_gib <= max_host_gib:
            raise ValueError("preferred_host_gib must be positive and <= max_host_gib")
        self.device = str(device)
        self.max_gib = max_gib
        self.reserve_mib = reserve_mib
        self.max_host_gib = max_host_gib
        self.preferred_host_gib = preferred_host_gib
        self.budget = None
        self.samples = []

    def __enter__(self):
        self("startup")
        if self.device.startswith("cuda"):
            index = int(self.device.split(":", 1)[1]) if ":" in self.device else 0
            self.budget = configure_gpu_budget(index, self.max_gib, self.reserve_mib)
        return self

    def __call__(self, phase: str):
        import psutil
        host = check_host_memory()
        process = psutil.Process()
        own = process.memory_info()
        rss = own.rss
        private = getattr(own, "private", 0)
        for child in process.children(recursive=True):
            try:
                memory = child.memory_info()
                rss += memory.rss
                private += getattr(memory, "private", 0)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        charged = max(rss, private)
        if charged >= self.max_host_gib * GIB:
            raise MemoryError(f"{phase}: max(process-tree RSS, private) exceeds {self.max_host_gib} GiB")
        sample = {"phase": phase, **host, "process_tree_rss_bytes": rss,
                  "process_tree_private_bytes": private if platform.system() == "Windows" else None,
                  "host_charged_bytes": charged,
                  "above_preferred_host": charged >= self.preferred_host_gib * GIB}
        if self.budget is not None:
            sample["gpu"] = check_gpu_headroom(self.budget)
        self.samples.append(sample)
        return None

    def check_device(self, phase: str):
        """Check a device-only transfer between full host budget checks."""
        if self.budget is None:
            raise RuntimeError('Device-only guard requires an active CUDA budget')
        self.samples.append({'phase': phase, 'gpu': check_gpu_headroom(self.budget)})

    def __exit__(self, exc_type, exc_value, tb):
        if exc_type is None:
            self("complete")
        return False


def ffmpeg_path() -> str:
    path = shutil.which("ffmpeg")
    if path:
        return path
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def platform_info() -> dict:
    import sys
    from importlib.metadata import PackageNotFoundError, version
    packages = {}
    for name in ("torch", "torchvision", "nvidia-cutlass-dsl", "cuda-python", "diffusers",
                 "transformers", "accelerate", "safetensors", "imageio", "imageio-ffmpeg", "av"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {"python": sys.executable, "python_version": platform.python_version(),
            "platform": platform.platform(), "packages": packages,
            "uv_cache_dir": os.environ.get("UV_CACHE_DIR")}
