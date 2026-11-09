# FastH3 RTX50

[English](README.md)

在 NVIDIA RTX 50 系列显卡上运行 MiniMax H3，本地生成带立体声音轨的视频。
四种生成任务共用一个命令行入口，模型权重在使用者本地下载和转换。

## Windows 安装

安装 [uv](https://docs.astral.sh/uv/getting-started/installation/) 和兼容 CUDA 12.8 的
NVIDIA 驱动。在仓库根目录执行。

```powershell
powershell -ExecutionPolicy Bypass -File scripts/bootstrap-win.ps1
.\.venv-win\Scripts\Activate.ps1
fasth3 doctor --gpu
```

Windows x64 锁定 Python 3.12.13 与 CUDA 12.8 版 PyTorch。Linux 和 WSL
可沿用相同 CLI，环境需安装对应平台的 CUDA PyTorch 和 CuTe DSL。
安装与测试细节见 [AGENTS.md](AGENTS.md)。

## 准备权重

先阅读 [MiniMax H3 社区许可证](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)，
再从上游 Hugging Face 仓库下载并转换固定版本。

```powershell
fasth3 prepare --models-dir models --task all --plan
fasth3 prepare --models-dir models --task all --accept-model-license
```

`--task f2va`、`--task fl2va` 和 `--task r2va` 使用同一套 Omni 权重，
`--task t2va` 只准备文生视频。
`--download-only` 下载并校验源文件，之后可以再转换。首次准备需要为源文件和转换产物
预留磁盘空间。模型目录存放在本地。

## 生成视频

示例使用 `examples/` 内的原创插图，按默认 768p 画布以 24 fps 导出视频。
FL2VA 示例使用 192 帧，确保指定的尾帧位于导出视频的结尾。

```powershell
fasth3 t2va --models-dir models --prompt "一颗橙色小球从左向右滚过青色地面。固定广角镜头，轻柔滚动声。" --num-frames 124 --seed 4277 --output outputs/text.mp4
fasth3 f2va --models-dir models --first-frame examples/first.png --prompt "橙色小球开始向右滚动。保持固定镜头。" --num-frames 124 --seed 4277 --output outputs/first.mp4
fasth3 fl2va --models-dir models --first-frame examples/first.png --last-frame examples/last.png --prompt "小球从起点滚到终点并停下。固定侧面镜头。" --num-frames 192 --seed 4277 --output outputs/first-last.mp4
fasth3 r2va --models-dir models --reference image:examples/first.png --prompt "参考橙色小球及平面插画风格。让小球缓缓滚过简洁的瓷砖地面。" --num-frames 124 --seed 4277 --output outputs/reference.mp4
```

T2VA 使用 FastH3 V2 文生视频权重。F2VA 和 FL2VA 使用 Omni 并分别提供首帧、
首尾帧条件。R2VA 支持按顺序传入多个 `--reference`，接受 `image:PATH`、
`video:PATH` 和 `audio:PATH`，至少提供一项视觉参考。
使用 `--prompt-file` 可以传入 UTF-8 文本。FL2VA 的帧数须满足 `17n + 5`，
在 120–360 帧范围内最多导出 345 帧。

`--debug-checks` 开启更深入的数值检查。R2VA 默认将参考图的短边调整为
256 像素。需要更多参考图细节时，可设置 `--reference-image-short-edge 768`。

设置环境变量 `DEEPSEEK_API_KEY` 后，加 `--pe` 启用 OpenH3-IR。
有视觉输入时，PE 会把参考图或少量采样帧发送给 DeepSeek，视频生成与模型权重保留在本机。

## 许可证

仓库原创代码和示例输入采用 [Apache-2.0](LICENSE)，再分发时保留[署名](NOTICE)。
上游模型、转换权重和模型输出适用各自条款，请遵守上游使用、地域及分发要求。
