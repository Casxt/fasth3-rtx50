# FastH3 RTX50

[简体中文](README.zh-CN.md)

FastH3 RTX50 adapts FastH3 video and stereo audio generation for systems with
limited RAM and VRAM. It supports NVIDIA RTX 50-series GPUs only. Components
use a mix of BF16 and MXFP8 to keep memory usage low while balancing speed and
output quality. The core kernels are written in CuTe DSL and can serve as a
reference for learning and adapting the implementation. The CLI prepares
upstream weights locally and runs all four generation modes.

Performance work so far has focused on the DiT. The VAE and Qwen conditioning
encoder have had limited tuning.

Main DiT block weights use one reusable GPU arena by default. For long sequences,
MXFP8 attention outputs stay on the GPU when the memory budget permits; a host
staging path is selected when the estimated workspace does not fit.

**We recommend using a coding agent to set up and run this project.** Give the
agent access to the repository and ask it to follow [AGENTS.md](AGENTS.md), which
covers dependencies, weight preparation, prompt enhancement, tests and
troubleshooting. A starting request is

```text
Read AGENTS.md. Check this machine and any existing environment and model files,
configure the pinned dependencies, prepare the weights for my chosen task, and
run the README example. Verify the output and report timing, Python process
memory, system memory and GPU memory separately.
```

## Install on Windows

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and an
NVIDIA driver compatible with CUDA 12.8. Run from the repository root.

```powershell
powershell -ExecutionPolicy Bypass -File scripts/bootstrap-win.ps1
.\.venv-win\Scripts\Activate.ps1
fasth3 doctor --gpu
```

The Windows x64 lock uses Python 3.12.13 and the CUDA 12.8 PyTorch wheels.
Linux and WSL use the same CLI. Use a platform-appropriate CUDA PyTorch and
CuTe DSL installation when adapting the environment. See [AGENTS.md](AGENTS.md)
for reproducible setup and test commands.

## Prepare weights

Read the [MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)
before using the model. Download and convert the pinned releases from the
original Hugging Face repositories.

```powershell
fasth3 prepare --models-dir models --task all --plan
fasth3 prepare --models-dir models --task all --accept-model-license
```

`--task t2va` and `--task fl2va` prepare the FastH3 T2VA text-to-video checkpoint.
`--task f2va` and `--task r2va` prepare FastH3 Omni. To prepare the alternative
FL2VA model, use `--task fl2va --model fasth3-omni-ref2va`.
`--download-only` verifies upstream files and leaves conversion for a later run.
The first complete preparation needs space for the source checkpoints and
converted weights. Keep the model directory local.

## Generate

The samples use the original illustration files under `examples/` and
export 124 frames at 24 fps on the default 768p canvas. The FL2VA demo
uses 192 frames to keep the closing frame at the end of the export.

```powershell
fasth3 t2va --models-dir models --prompt "A small orange ball rolls from left to right across a teal floor. Static wide shot. Soft rolling sound, no music." --num-frames 124 --seed 4277 --output outputs/text.mp4
fasth3 f2va --models-dir models --first-frame examples/first.png --prompt "The orange ball starts rolling to the right. Keep the camera static." --num-frames 124 --seed 4277 --output outputs/first.mp4
fasth3 fl2va --models-dir models --first-frame examples/first.png --last-frame examples/last.png --prompt "The ball rolls from its starting point to its final position and stops. Static side view." --num-frames 192 --seed 4277 --output outputs/first-last.mp4
fasth3 r2va --models-dir models --reference image:examples/first.png --prompt "Use the orange ball and flat illustration style as references. Show the ball rolling gently across a simple tiled floor." --num-frames 124 --seed 4277 --output outputs/reference.mp4
```

FL2VA defaults to the FastH3 T2VA text-to-video checkpoint. Add
`--model fasth3-omni-ref2va` to the `fl2va` command to
use the FastH3 Omni reference checkpoint with the same first and last frame
inputs. The two models can handle motion, detail, and endpoint transitions
differently from scene to scene. Test both with your own frames and prompt.

F2VA uses Omni with a first-frame condition. R2VA uses Omni and accepts ordered, repeated
`--reference image:PATH`, `video:PATH`, or `audio:PATH` values and requires at
least one visual reference. Use `--prompt-file` to supply exact UTF-8 prompt
text. FL2VA accepts frame counts of the form `17n + 5` in the 120–360
frame range, so its longest supported export is 345 frames.

Use `--debug-checks` for deeper numerical diagnostics. R2VA resizes image
references to a 256-pixel short edge by default. Use
`--reference-image-short-edge 768` when the reference needs more detail.

## Context IR and prompt quality

MiniMax's official H3-Context-IR is available as a hosted API and is outside
the open release. Final quality depends heavily on how the text, reference
media, timing, and dialogue are represented in the prompt. See
[MiniMax's H3 release notes](https://www.minimax.io/news/minimax-h3-open-source).

`--pe` is a demo integration of OpenH3-IR with the DeepSeek API. Set
`DEEPSEEK_API_KEY` in the process environment. With visual inputs, PE sends
a bounded set of reference images or sampled video frames to DeepSeek.
Generation and model weights stay local.
[OpenH3-IR project](https://github.com/ruashots/open-h3-ir)

For production work, use a language model of your choice to develop and tune
your own structured prompts. Review subject bindings, shot timing and dialogue,
then compare renders with a fixed seed and pass the chosen prompt via
`--prompt-file`.

## Known quality limitations

T2VA can produce patchy blur, particularly on faces and hair. The severity
varies across generations, so quality is partly down to the seed and individual
render.

F2VA and FL2VA can show mild blur when run with the FastH3 T2VA checkpoint. In a
matched 5-second F2VA test with that checkpoint, switching the DiT from MXFP8 to
BF16 made no visible improvement. The Omni reference checkpoint used by R2VA
avoided this type of blur in our frame-control samples. F2VA already uses Omni.
For FL2VA, select it with `--model fasth3-omni-ref2va`.

## Demos and benchmark results

See [demo sources and usage conditions](examples/demos/README.md) for character
credits and the terms that apply separately from the code license.

Windows, RTX 5080 16 GB, 1344×768 at 24 fps, a 10 GiB PyTorch allocator cap,
and a 256 MiB device memory reserve. Each row is one measured run. Generation
includes local inference and export. End-to-end also includes OpenH3-IR and any
packaging in that run. Rows using an existing rewritten prompt report generation
time only and show a dash in the end-to-end column.
Python memory lists process-tree working set / private
commit. System memory lists physical used (total minus available) / system commit.
Each host figure is a separate sampled peak during generation, in GiB.
CUDA columns show peak allocation and sampled peak reservation. Demo MP4s use
H.265 CRF 20. The release transcode is excluded from the timings.

| Mode | Input | Output | Generation (s) | End to end (s) | Python memory (GiB) | System memory (GiB) | CUDA allocated / reserved (GiB) | Demo |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| T2VA | Text only | 5 s | 230.8 | 270.2 | 8.78 / 22.74 | 27.57 / 48.57 | 3.65 / 4.52 | [MP4](examples/demos/t2va-5s.mp4) |
| T2VA | Text only | 10 s | 460.3 | 496.0 | 9.07 / 23.06 | 27.96 / 48.94 | 3.85 / 6.10 | [MP4](examples/demos/t2va-10s.mp4) |
| T2VA | Text only | 15 s | 720.2 | 745.0 | 9.47 / 23.52 | 28.38 / 49.92 | 5.15 / 7.65 | [MP4](examples/demos/t2va-15s.mp4) |
| F2VA | 1 first-frame image | 5 s | 236.5 | 278.4 | 16.86 / 28.42 | 35.50 / 54.51 | 3.66 / 4.52 | [MP4](examples/demos/f2va-5s.mp4) |
| F2VA | 1 first-frame image | 10 s | 414.3 | 448.7 | 16.86 / 28.81 | 36.00 / 55.18 | 3.70 / 5.78 | [MP4](examples/demos/f2va-10s.mp4) |
| F2VA | 1 first-frame image | 15 s | 673.7 | 705.1 | 16.86 / 29.27 | 35.70 / 55.76 | 4.77 / 7.34 | [MP4](examples/demos/f2va-15s.mp4) |
| FL2VA (Omni) | 2 images (first and last frames) | 5 s | 246.1 | 321.4 | 16.91 / 28.52 | 36.09 / 54.84 | 3.67 / 4.54 | [MP4](examples/demos/fl2va-5s.mp4) |
| FL2VA (Omni) | 2 images (first and last frames) | 10 s | 456.3 | 518.6 | 16.95 / 28.89 | 36.13 / 55.03 | 3.71 / 5.92 | [MP4](examples/demos/fl2va-10s.mp4) |
| FL2VA (Omni) | 2 images (first and last frames) | 15 s | 769.4 | 855.7 | 16.91 / 29.24 | 36.30 / 54.95 | 4.86 / 7.46 | [MP4](examples/demos/fl2va-15s.mp4) |
| FL2VA (T2VA) | 2 images (first and last frames) | 5 s | 266.0 | — | 16.92 / 28.32 | 37.39 / 56.17 | 3.67 / 4.67 | [MP4](examples/demos/fl2va-t2va-5s.mp4) |
| FL2VA (T2VA) | 2 images (first and last frames) | 10 s | 527.9 | — | 16.91 / 28.69 | 37.46 / 56.23 | 4.01 / 6.35 | [MP4](examples/demos/fl2va-t2va-10s.mp4) |
| FL2VA (T2VA) | 2 images (first and last frames) | 15 s | 842.4 | — | 16.91 / 29.08 | 37.39 / 56.82 | 5.32 / 7.92 | [MP4](examples/demos/fl2va-t2va-15s.mp4) |
| R2VA | 1 reference image + 1 × 5 s video | 5 s | 496.3 | — | 17.41 / 28.92 | 34.93 / 53.26 | 3.73 / 6.38 | [MP4](examples/demos/r2va-5s.mp4) |
| R2VA | 1 reference image + 1 × 10 s video | 10 s | 1344.3 | — | 17.90 / 29.80 | 36.18 / 54.10 | 6.64 / 7.88 | [MP4](examples/demos/r2va-10s.mp4) |
| R2VA | 1 reference image + 1 × 15 s video | 15 s | 2258.0 | — | 18.34 / 30.64 | 35.76 / 54.76 | 8.49 / 8.74 | [MP4](examples/demos/r2va-15s.mp4) |

R2VA runtime and memory depend on both input and output duration. The three
character-replacement cases use the 0–5, 5–15 and 15–30 second segments of
the short film, with equal input and output lengths and one Kimi identity image
resized to a 512-pixel short edge. The three 768p R2VA rows were rerun with
v0.1.1 defaults and all eight denoising steps, reusing the rewritten prompts.
Generation includes inference, VAE decoding and export, and excludes PE and the
H.265 CRF 20 transcode. Other modes and 480p results remain historical and were
not rerun for v0.1.1.
These R2VA runs use the production CLI directly: requested exports of
120 / 240 / 360 frames are automatically aligned internally to 124 / 243 / 362
frames and then exported at the requested count.
NVML sampled whole-board GPU memory peaks for the 5 / 10 / 15 s R2VA runs were
9.41 / 10.69 / 11.35 GiB, respectively. These include driver and other-process usage
and differ from the CUDA allocated / reserved figures in the table.
Historical FL2VA clips were conformed to 120, 240 and 360 frames while preserving
both endpoints. The historical 15 s FL2VA samples used a local wrapper to generate
362 aligned frames before conforming the export.

**Chained short film** · [The Spare Moon (30 s)](examples/demos/the-spare-moon-30s.mp4) ·
The last frame of a 15 s F2VA render starts a 15 s FL2VA continuation.
Combined generation took 1442.0 s and end-to-end took 1555.8 s. Python memory
peaked at 16.93 / 29.27 GiB, system memory at 36.20 / 55.76 GiB, and CUDA
allocation / reservation at 4.88 / 7.58 GiB.

### 480p speed tests

Both runs produce 5 s at 832×480 and 24 fps and reuse the prompts from the
corresponding 768p samples. Timings include generation and export.

| Mode | Input | Generation (s) | Python memory (GiB) | System memory (GiB) | CUDA allocated / reserved (GiB) |
| --- | --- | ---: | ---: | ---: | ---: |
| T2VA | Text only | 149.2 | 8.50 / 22.01 | 24.32 / 42.83 | 3.52 / 4.03 |
| R2VA | 1 reference image + 1 × 5 s video | 186.8 | 17.08 / 27.90 | 32.95 / 48.71 | 3.56 / 4.09 |

## License

Original repository code and the generic orange-ball inputs
[`examples/first.png`](examples/first.png) and [`examples/last.png`](examples/last.png)
use [Apache-2.0](LICENSE) with [attribution notices](NOTICE).
Character designs and character-based demo media are outside that code license;
see [demo sources and usage conditions](examples/demos/README.md).
Upstream model assets, converted weights and outputs remain governed by their
applicable terms, including usage, territory and distribution requirements.
