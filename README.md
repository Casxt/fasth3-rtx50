# FastH3 RTX50

[简体中文](README.zh-CN.md)

Local MiniMax H3 video and stereo audio generation for NVIDIA RTX 50 GPUs.
The CLI prepares upstream weights locally and runs four generation modes from
one entry point.

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

Use `--task r2va`, `f2va`, or `fl2va` for the shared Omni assets, or
`--task t2va` for text generation. `--download-only` verifies upstream files and leaves
conversion for a later run. The first complete preparation needs space for the
source checkpoints and converted weights. Keep the model directory local.

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

T2VA uses the FastH3 V2 text checkpoint. F2VA and FL2VA use Omni with
first-frame or first/last-frame conditions. R2VA accepts ordered, repeated
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
render. Decoding the same latents in BF16 and MXFP8 showed the same visible
patterns in two affected clips.

## Previous benchmark results

Earlier 768p (1344×768), 24 fps demo runs at `6a5a30c` on Windows with an RTX 5080
16 GB, a 10 GiB PyTorch allocator cap, and a 256 MiB device memory reserve. These are
single-run historical measurements. Generation time covers local inference
and export. End-to-end time also includes OpenH3-IR and any demo packaging.
Host memory is the sampled process-tree peak. CUDA columns show the allocation
peak and sampled reservation peak in GiB.

| Mode | Output | Generation (s) | End to end (s) | Host peak (GiB) | CUDA allocated / reserved (GiB) | Demo MP4 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| T2VA | 5 s | 230.8 | 270.2 | 22.74 | 3.65 / 4.52 | |
| T2VA | 10 s | 460.3 | 496.0 | 23.06 | 3.85 / 6.10 | |
| T2VA | 15 s | 720.2 | 745.0 | 23.52 | 5.15 / 7.65 | |
| F2VA | 5 s | 236.5 | 278.4 | 28.42 | 3.66 / 4.52 | |
| F2VA | 10 s | 414.3 | 448.7 | 28.81 | 3.70 / 5.78 | |
| F2VA | 15 s | 673.7 | 705.1 | 29.27 | 4.77 / 7.34 | |
| FL2VA | 5 s | 246.1 | 321.4 | 28.52 | 3.67 / 4.54 | |
| FL2VA | 10 s | 456.3 | 518.6 | 28.89 | 3.71 / 5.92 | |
| FL2VA | 15 s | 769.4 | 855.7 | 29.24 | 4.86 / 7.46 | |
| R2VA | 5 s | 513.7 | 603.6 | 28.93 | 3.74 / 6.36 | |
| R2VA | 10 s | 1464.6 | 1524.2 | 30.22 | 5.56 / 9.29 | |
| R2VA | 15 s | 2605.6 | 2746.9 | 31.09 | 7.88 / 8.73 | |

R2VA used video references of the same nominal duration as the output. The
reference encoder consumed 107, 226, and 345 frames after alignment. The
FL2VA demo runs rendered 124, 243, and 362 aligned frames, then conformed
them to 120, 240, and 360 frames while preserving both endpoints. The
362-frame run used a local demo wrapper.

## License

Original repository code and demo inputs use [Apache-2.0](LICENSE) with
[attribution notices](NOTICE). Upstream model assets, converted weights and
outputs are governed by their applicable terms. Read the upstream agreement
and observe its usage, territory and distribution requirements.
