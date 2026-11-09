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

Pass `--pe` to use OpenH3-IR with the DeepSeek API. Set `DEEPSEEK_API_KEY`
in the process environment. With visual inputs, this sends bounded reference
images or video samples to DeepSeek. Generation and model weights stay local.

## License

Original repository code and demo inputs use [Apache-2.0](LICENSE) with
[attribution notices](NOTICE). Upstream model assets, converted weights and
outputs are governed by their applicable terms. Read the upstream agreement
and observe its usage, territory and distribution requirements.
