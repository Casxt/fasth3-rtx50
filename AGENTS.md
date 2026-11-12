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
powershell -ExecutionPolicy Bypass -File scripts/bootstrap-win.ps1
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

For Linux or WSL, install the platform's CUDA PyTorch wheels and the dependencies
and `pe,test` extras from `pyproject.toml` in a Python 3.12 environment. The Windows
lock applies to Windows. Run `fasth3 doctor --gpu` after setup. Keep WSL model
files on its Linux filesystem for local I/O.

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
in Git and linked from both READMEs. Keep checkpoints, derived weights,
benchmark reports, logs, local research notes, credentials and downloaded
reference media outside the release tree. Treat the model directory as local
user data. Pin and verify external dependencies and the model manifest before
changing any revision.
