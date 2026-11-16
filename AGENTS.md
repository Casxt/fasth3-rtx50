# FastH3 agent setup

Read this file and the relevant README before configuring or running the project.
Use the user's existing environment and model directory when compatible. Run
commands from the repository root and use the environment's Python executable.

## Install and verify

The runtime requires an NVIDIA RTX 50-series CUDA SM120 GPU. The tested Windows
x64 environment uses Python 3.12.13 and CUDA 12.8 PyTorch. Inspect `nvidia-smi`
and available host memory and disk space before preparing or running models.

On Windows, install uv and run

```powershell
uv run --no-project --python 3.12.13 scripts/bootstrap.py
.\.venv-win\Scripts\fasth3.exe doctor --gpu
```

The bootstrap script syncs `requirements-win.lock` with package hashes and
installs this repository in editable mode. The lock includes pytest and
OpenH3-IR. Direct executable paths work without activating the environment.
The script rejects an existing `.venv-win` with a different Python version.

If Python works but `fasth3.exe` is missing, install the package with
`uv pip install --python .\.venv-win\Scripts\python.exe --no-deps --editable .`.
For a source-tree check, the equivalent entry point is
`.\.venv-win\Scripts\python.exe -m fasth3_pipeline.cli`.

For Linux x86_64 or WSL, follow the cache exports and
`uv run --no-project --python 3.12 scripts/bootstrap.py` example in
[README.md](README.md#install-on-linux--wsl).
The PowerShell and Bash launchers delegate installation to this shared Python
entry; `python scripts/bootstrap.py --dry-run` previews its plan without setup.
Use Python 3.12; the checked Linux environment uses 3.12.3, rather than requiring
the Windows patch version. `FASTH3_WORK_DIR` selects a work directory with room
for the venv, caches and models. `FASTH3_PYTHON` and `FASTH3_PYPI_INDEX` select
the interpreter and PyPI index. Keep WSL model files on its Linux filesystem.

The bootstrap installs `requirements-linux.txt`, including the CUDA stack and
`pe,test` dependencies, then this source tree in editable mode. There is one
dependency file per platform: `requirements-linux.txt` and `requirements-win.lock`.
The Linux file pins official CUDA 12.8 PyTorch wheel URLs; the other packages
use the selected PyPI index. It uses Torch 2.10.0+cu128, torchvision 0.25.0+cu128, CuTe DSL
4.8.1 and the pinned Diffusers archive from `pyproject.toml`. Linux
cuda-python/cuda-bindings must use 12.9.4 because the Linux Torch wheel requires
that binding version; the Windows cuda-python pin remains 12.9.10. Do not force
12.9.10 onto Linux by bypassing dependency checks. The Linux snapshot is not a
complete wheel-hash lock. Before consolidating the dependency files, the
two-stage bootstrap was run in a new `.venv-linux`
with the existing dependency cache: installation took 16.789 s, or 36.297 s
including package compatibility checks, CPU/GPU doctor and three actual CuTe
kernel checks. The full Linux CPU suite recorded 403 passed / 51 skipped in
4.69 s; the full GPU suite recorded 454 passed / zero skipped in 39.64 s,
including real NVFP4 checkpoint probes under the 10 GiB allocator cap.

Keep UV/PIP/Hugging Face/Torch, CUDA, Triton and CuTe caches and temporary files
on the chosen work disk. CuTe DSL 4.8.1 uses `CUTE_DSL_CACHE_DIR`. Bootstrap
exports affect only its subprocesses, so retain the README cache exports in
subsequent run shells. The Linux CLI defaults OMP/MKL threads to
`min(8, effective_cpu_count())` before importing Torch and preserves explicit
settings; bootstrap gives RAYON the same default. Do not use host CPU count
alone to size thread pools in a container.

Run the environment's `uv pip check`, `fasth3 --help` and `fasth3 doctor` after
setup. Run `fasth3 doctor --gpu` with an idle GPU; it checks the device rather
than executing project kernels. Coordinate any targeted CUDA tests separately
from conversion or inference. Measured generation coverage and input conditions
are documented in [benchmarks/linux-validation.md](benchmarks/linux-validation.md).

`doctor` reports the optional `open-h3-ir` version but can succeed when it is
missing. Before PE, explicitly check it with the same environment

```powershell
.\.venv-win\Scripts\python.exe -c "import h3ir; from importlib.metadata import version; print(version('open-h3-ir'))"
```

If a locked package is missing from an older environment, rerun the bootstrap.
Use the versions in the lock and package metadata when resolving import errors.

## Prepare weights

Read the model agreement linked in the README. Respect any acknowledgement
already supplied by the user in this session. Inspect the download plan before
preparing the selected task

```powershell
.\.venv-win\Scripts\fasth3.exe prepare --models-dir models --task fl2va --plan
.\.venv-win\Scripts\fasth3.exe prepare --models-dir models --task fl2va --accept-model-license
```

`--plan` queries upstream metadata. `--download-only` downloads and verifies
source assets while deferring conversion. `--task all` prepares both checkpoint
families and shared components. The preparation command downloads pinned source
revisions and converts the DiT and VAE overlays locally. Source manifests live
in `fasth3_pipeline/prepare_weights.py`.

| Task | Default DiT source | Inputs |
| --- | --- | --- |
| `t2va` | FastH3 T2VA text-to-video | Text |
| `f2va` | FastH3 Omni reference-to-video | First-frame image |
| `fl2va` | FastH3 T2VA text-to-video | First- and last-frame images |
| `r2va` | FastH3 Omni reference-to-video | Ordered visual and optional audio references |

For the Omni FL2VA alternative, add `--model fasth3-omni-ref2va` to both the
task's preparation command and generation command. Name the checkpoint family
explicitly when reporting results. Frame-control quality varies by source.

Under the model directory, `bf16` and `mxfp8-packed` hold the Omni source and
converted DiT. `t2va-v2-bf16` and `t2va-v2-mxfp8-packed` hold the T2VA source and
converted DiT. `encoder-nvfp4` holds Qwen, and `shared` holds tokenizers and VAEs.
Use `prepare` to build missing assets. Preserve user-owned checkpoints and
verified download fragments when repairing a preparation failure.

## Prompt enhancement and API

`--pe` uses OpenH3-IR 0.4.1 with DeepSeek Flash. Defaults are
`https://api.deepseek.com`, `deepseek-flash`, a 120-second request timeout,
visual reference sampling and thinking enabled. OpenH3-IR is a dependency
installed by the lock or `pe` extra. Keep `DEEPSEEK_API_KEY` in the process
environment. Never place credentials in the workspace, logs, test fixtures or
commits. PE sends bounded reference images or video frames to the provider.
Use assets the user has authorized for that provider.

Review a compiled prompt before running generation with the weights-free tool.

```powershell
.\.venv-win\Scripts\python.exe tools/rewrite_prompt.py r2va --reference image:examples/first.png --prompt "Use the orange ball as a reference. Show it rolling gently to the right." --num-frames 192 --output outputs/review.prompt.txt
```

The tool also accepts `--pe-model`, `--pe-base-url` and `--pe-timeout`. Its receipt
is `outputs/review.prompt.txt.pe.json`. Check its PE source and findings, then
review subject identities, replacement targets, left/right positions, dialogue
languages and shot timing. Pass the reviewed text with `--prompt-file`.
The README describes the limits of the OpenH3-IR demo integration.

## Run and verify

Use the four generation examples in [README.md](README.md). The released CLI
uses the 768p canvas at 24 fps and performs eight DiT forwards. Frame counts are
120–360. FL2VA additionally requires `17n + 5`, so 124, 192 and 345 are valid.
The longest public FL2VA export is 345 frames. Historical 480p and 362-frame
benchmarks used local wrappers.

R2VA accepts repeated `--reference image:PATH`, `video:PATH` and `audio:PATH` in
the prompt's reference order and requires a visual input. Image short edge
defaults to 256. Use `--reference-image-short-edge` when a different image size
is needed, and record it with the input video duration and resolution.

Run one GPU-heavy command at a time. The CLI enforces a 10 GiB PyTorch allocator
cap and at least 256 MiB free board memory. Report budget failures and inspect
the recorded phase. Use `--debug-checks` for suspected numerical or weight
problems.

The main DiT blocks reuse one fixed weight arena by default. Long requests use
2048-token QKV chunks without an input quantization cache. When the planned
workspace fits, fixed MXFP8 attention output buffers stay on the GPU; otherwise
the request uses host staging. Inspect `streaming.dit_policy`, including
`attention_output_runtime` when allocation caused a fallback.

During DiT, the runtime may lower the allocator cap to target 11.5 GiB of total
board usage, subtracting a single `nvidia-smi` sample of external occupancy and
an additional 128 MiB margin. If that query is unavailable, CUDA memory accounting
is used. The receipt is `streaming.dit_budget`; its `source` identifies the
measurement and `restored` confirms restoration before VAE decoding. The original
allocator limit and the 256 MiB board reserve still apply.

Each generation produces an MP4 and a same-stem JSON report. Output paths must
be unused. Check `status`, `transformer_forwards`, model source, dimensions,
frame count, fps and audio duration. Keep prompt, reference order, seed and
sampling settings fixed in comparisons. Separate generation time from PE and
other packaging time.

The report's `guard_samples` distinguish Python process-tree working set
(`process_tree_rss_bytes`) and private commit (`process_tree_private_bytes`)
from system available physical memory (`available_bytes`) and system commit
(`commit_bytes`). Record system physical total as well when computing physical
used as total minus available. GPU records distinguish allocation, reservation,
peak allocation and free board memory. Label sampled peaks and memory units.

On Linux, report Python process-tree RSS separately from system or container
memory; private commit and system commit are Windows-specific fields. Container
`psutil` totals can describe the host. Use `available_scope`, cgroup limits and
headroom, and the separately named `host_*` counters rather than treating host
RAM as the instance allocation. The guard checks visible cgroup v1/v2 ancestors
and credits reclaimable inactive file cache when estimating headroom; CPU
capacity combines quota and affinity. Do not infer Python memory from
container usage, which also includes file cache and other processes.

## Tests

Run `.\.venv-win\Scripts\python.exe -m pytest tests -q` for the CPU suite. For
CUDA checks, set `$env:FASTH3_TEST_CUDA = '1'` and run the relevant tests while
the GPU is idle. Offload event-order tests additionally use
`$env:FASTH3_RUN_GPU_TESTS = '1'`. PE transport tests can run with mocked requests before using
the live API. Select tests for the components changed.

## Release contents

Publish owned source, package metadata, install locks, tests, the two READMEs,
these instructions, the license and notices, demo commands and owned demo
inputs. Approved H.265 CRF 20 samples under `examples/demos/` are tracked directly
in Git. Keep sample links in both READMEs on GitHub’s `/raw/refs/heads/main/`
route so private-repository downloads use the reader’s GitHub session. Do not
commit expiring raw-access tokens or use unauthenticated raw-host URLs.
The selected Linux T2VA 5 s and Windows R2VA 10 s samples also use GitHub-native
video attachments in both READMEs. Other Linux proof videos may remain tracked
without README media links. Curated, sanitized benchmark summaries and
reproduction recipes may also be tracked, with input conditions, provenance and the scope of validation made
explicit. Keep checkpoints, derived weights, raw logs and traces, research
notes, private configurations, credentials and downloaded reference media
outside the release tree. Treat the model directory as local user data. Pin
and verify external dependencies and the model manifest before changing any
revision.
