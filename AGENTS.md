# FastH3 agent setup

## Install and verify

From the repository root on Windows x64, install uv and run
`powershell -ExecutionPolicy Bypass -File scripts/bootstrap-win.ps1`.
The script syncs the Windows lock and installs the CLI. Invoke
`.\.venv-win\Scripts\fasth3.exe doctor --gpu` when the GPU is idle.

The lock includes the OpenH3-IR version used by PE. In a separate development
environment, install the package with the `pe` extra after installing the
platform's CUDA PyTorch wheels. Run `fasth3 doctor` to check the installed
packages. Keep `DEEPSEEK_API_KEY` in the process environment. Never place
credentials in the workspace or include them in logs, test fixtures or commits.

## Prepare and test

1. Read the upstream model agreement. Run `fasth3 prepare --models-dir models --task all --plan` to inspect the sources. Run the same command with `--accept-model-license` in place of `--plan` to download and convert. Use `--download-only` to stage and verify without CUDA.
2. Run `.\.venv-win\Scripts\python.exe -m pytest tests -q` for the CPU suite. To include the CUDA checks, set `$env:FASTH3_TEST_CUDA = '1'` in a clean PowerShell session and repeat the same command while the GPU is idle. Run only one GPU-heavy command at a time.
3. Use the four task examples in [README.md](README.md) for real smoke runs. Generation should produce an MP4 with audio and matching frame counts, plus a JSON report. Keep inputs fixed when comparing repeat runs.
4. To exercise PE, use `--pe` with a test asset you can send to the configured provider. Test with mocked transport first. Check generated time bounds against the exported frame count and keep reference order fixed.

Review a compiled prompt before running generation with the weights-free tool.

```powershell
.\.venv-win\Scripts\python.exe tools/rewrite_prompt.py r2va --reference image:examples/first.png --prompt "Use the orange ball as a reference. Show it rolling gently to the right." --num-frames 192 --output outputs/review.prompt.txt
```

## Release contents

Publish owned source, package metadata, install locks, tests, the two READMEs,
these instructions, the license and notices, demo commands and owned demo
inputs. Keep checkpoints, derived weights, benchmark reports, logs, local
research notes, credentials and downloaded reference media outside the release
tree. Treat the model directory as local user data. Pin and verify external
dependencies and the model manifest before changing any revision.
