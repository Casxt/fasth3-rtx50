# CPU/GPU transfer baseline

Measured on 2026-11-16 (Asia/Shanghai), using
[`tools/benchmark_transfers.py`](../tools/benchmark_transfers.py). Full results
and numeric metadata are in [transfer-baseline.json](transfer-baseline.json).
Both GPUs were idle before their runs. Windows desktop applications remained
open; Linux ran in an exclusive GPU window between model requests. No game,
model inference or conversion ran concurrently with these measurements.

## Hardware and conditions

| | Windows | Linux container |
| --- | --- | --- |
| GPU | RTX 5080, 16 GiB class | RTX 5090, 32 GiB class |
| CPU observed live | Ryzen 9 9950X | Xeon Gold 6459C |
| OS / Python | Windows 11 / 3.12.13 | Ubuntu 22.04 userspace, Linux 5.15 / 3.12.3 |
| Torch / CUDA runtime | 2.10.0+cu128 / 12.8 | 2.10.0+cu128 / 12.8 |
| NVIDIA driver | 621.14 | 595.71.05 |
| Host-visible logical CPUs / process affinity count | 32 / 32 | 128 / 128 |
| Effective CPU capacity / benchmark Torch threads | 32 / 8 | 16 / 8 |
| CPU quota | Not applicable | 1,600,000 / 100,000 us = 16 CPUs |
| Host-visible physical RAM | 93.38 GiB | 754.54 GiB |
| Container RAM limit | Not applicable | 92 GiB |
| PCIe maximum and under-copy-load state | Gen5 x16 | Gen5 x16 |

PCIe state was sampled three times during a separate sustained H2D/D2H loop,
outside timed samples; all six load samples reported Gen5 x16. Before-run
samples also reported Gen5 x16. The Linux GPU's `nvidia-smi topo -m` CPU affinity
is `32-63,96-127`, with NUMA affinity 1. The container does not expose this GPU's
sysfs `numa_node` or `local_cpulist`; no NUMA policy or process affinity was
changed. A later CPU-only cross-check confirmed two sockets, 32 cores per
socket, two threads per core, online/affinity CPUs `0-127`, NUMA node 0 CPUs
`0-31,64-95` and NUMA node 1 CPUs `32-63,96-127`. Host-visible RAM and CPU counts
describe the host, not the container's
allocation. Earlier instance descriptions must not replace these live readings.

## 256 MiB results

Median transfer rates. **GB/s is decimal; GiB/s is binary.** CPU wall time includes
submission and completion wait. CUDA events measure the stream interval; for
pageable memory that interval can include staging/submission gaps and must not
be described as pure DMA time.

| Platform | Host buffer | Copy | Wall GB/s | Wall GiB/s | CUDA-event GB/s | Wall ms |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Windows RTX 5080 | pageable | H2D | 20.73 | 19.31 | 20.77 | 12.949 |
| Windows RTX 5080 | pageable | D2H | 19.60 | 18.25 | 19.63 | 13.698 |
| Windows RTX 5080 | pinned | H2D | 56.44 | 52.57 | 56.92 | 4.756 |
| Windows RTX 5080 | pinned | D2H | 56.89 | 52.98 | 57.33 | 4.719 |
| Linux RTX 5090 | pageable | H2D | 12.37 | 11.52 | 12.37 | 21.708 |
| Linux RTX 5090 | pageable | D2H | 22.15 | 20.63 | 22.17 | 12.120 |
| Linux RTX 5090 | pinned | H2D | 56.32 | 52.46 | 56.44 | 4.766 |
| Linux RTX 5090 | pinned | D2H | 57.15 | 53.23 | 57.26 | 4.697 |

Pinned 256 MiB transfers reach approximately 56–57 GB/s on both machines.
These measurements provide no evidence that the Linux GPU has a degraded PCIe
link. Linux pageable H2D is slower than Windows in this run, despite essentially
identical large pinned transfer rates. CPU staging, memory placement and the
host-memory path are worth checking in the application; this isolated test does
not establish the cause of model bubbles or show that offload copies overlap.

## Small-copy latency

Pinned 1 MiB copies expose the cost of submitting and waiting for short transfers.
The wall/event difference is a diagnostic observation, not a separate exact CPU
staging measurement.

| Platform | Copy | Wall us | CUDA-event us |
| --- | --- | ---: | ---: |
| Windows RTX 5080 | H2D | 66.50 | 32.18 |
| Windows RTX 5080 | D2H | 55.15 | 20.99 |
| Linux RTX 5090 | H2D | 52.83 | 44.30 |
| Linux RTX 5090 | D2H | 29.18 | 20.83 |

For example, the Linux pinned D2H wall rate increases from about 35.94 GB/s at
1 MiB to 57.15 GB/s at 256 MiB. Combining genuinely small transfers can reduce
submission overhead; it does not increase the already strong large-copy link
rate. Runtime changes still require an application-level comparison.

## Method and reproduction

The same timed code path ran on both platforms: 1, 16, 64 and 256 MiB, H2D and
D2H, pageable and pinned, three warmups and 20 individually synchronized samples
per case. All 32 cases passed full-payload verification. Buffers are allocated
and pages touched before timing. Event handles are warmed before measurements.
Verification and NVIDIA telemetry are outside timing. The separate PCIe load
probe is excluded from reported transfer latency and bandwidth.

Each run preallocates 256 MiB on the GPU and 512 MiB on the host. Verification
uses temporary storage; PyTorch peak allocation was about 512 MiB on each GPU.
The benchmark uses the project's 10 GiB allocator cap and at least 256 MiB board
reserve. These limits are ceilings, not amounts of memory allocated. Memory
counters in the JSON describe process/system/container conditions at sampling,
not an inference memory peak; Linux container usage also includes existing file
cache and other processes.

```bash
python tools/benchmark_transfers.py --output transfer.json
```

Use the project's Python environment on either platform. Run one GPU benchmark
at a time, retain the condition note, and compare both wall and event results.
These are repeated copies of reused buffers, not cold weight reads or a
measurement of storage-to-GPU throughput. No application-level offload overlap
is claimed.

The tool supports `--condition`, `--sizes-mib`, `--repeats` and `--warmup` for
reproduction. It does not download models or change affinity/NUMA policy.
