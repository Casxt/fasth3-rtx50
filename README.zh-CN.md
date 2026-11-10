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

## Context IR 与提示词质量

MiniMax 的官方 H3-Context-IR 未随模型开源，目前以托管 API 提供。
实际画质和指令遵循效果，很大程度取决于提示词如何组织文字、参考素材、镜头时序和对白。
详见 [MiniMax H3 发布说明](https://www.minimax.io/news/minimax-h3-open-source)。

`--pe` 提供 OpenH3-IR 与 DeepSeek API 的演示集成。
使用前设置环境变量 `DEEPSEEK_API_KEY`。有视觉输入时，PE 会向 DeepSeek 发送限量的
参考图或视频采样帧，视频生成与模型权重留在本机。
[OpenH3-IR 项目](https://github.com/ruashots/open-h3-ir)

实际使用建议用自己选的大模型调试结构化提示词，重点核对人物与参考素材的绑定、
镜头时序和对白。固定 seed 做成片对照，再通过 `--prompt-file` 传入调好的版本。

## 已知画质限制

T2VA 有时会出现块状模糊，在脸和头发上较明显。程度随 seed 和单次生成而变化，
需要挑片。

## 历史性能数据

旧版样片在 `6a5a30c`、Windows、RTX 5080 16 GB 上测试，输出为 1344×768、24 fps，
PyTorch 分配上限 10 GiB，整卡显存至少预留 256 MiB。每行是一次历史实测。
生成耗时含本地推理和导出，全流程另含 OpenH3-IR 和样片封装。
主机内存为进程树采样峰值，CUDA 两项数值分别为分配峰值和保留显存采样峰值，单位均为 GiB。

| 模式 | 输入 | 输出时长 | 生成（秒） | 全流程（秒） | 主机峰值（GiB） | CUDA 分配 / 保留（GiB） | Demo MP4 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| T2VA | 仅文本 | 5 秒 | 230.8 | 270.2 | 22.74 | 3.65 / 4.52 | |
| T2VA | 仅文本 | 10 秒 | 460.3 | 496.0 | 23.06 | 3.85 / 6.10 | |
| T2VA | 仅文本 | 15 秒 | 720.2 | 745.0 | 23.52 | 5.15 / 7.65 | |
| F2VA | 1 张首帧图 | 5 秒 | 236.5 | 278.4 | 28.42 | 3.66 / 4.52 | |
| F2VA | 1 张首帧图 | 10 秒 | 414.3 | 448.7 | 28.81 | 3.70 / 5.78 | |
| F2VA | 1 张首帧图 | 15 秒 | 673.7 | 705.1 | 29.27 | 4.77 / 7.34 | |
| FL2VA | 2 张图（首帧和尾帧） | 5 秒 | 246.1 | 321.4 | 28.52 | 3.67 / 4.54 | |
| FL2VA | 2 张图（首帧和尾帧） | 10 秒 | 456.3 | 518.6 | 28.89 | 3.71 / 5.92 | |
| FL2VA | 2 张图（首帧和尾帧） | 15 秒 | 769.4 | 855.7 | 29.24 | 4.86 / 7.46 | |
| R2VA | 1 张参考图 + 1 段 5 秒视频 | 5 秒 | 513.7 | 603.6 | 28.93 | 3.74 / 6.36 | |
| R2VA | 1 张参考图 + 1 段 10 秒视频 | 10 秒 | 1464.6 | 1524.2 | 30.22 | 5.56 / 9.29 | |
| R2VA | 1 张参考图 + 1 段 15 秒视频 | 15 秒 | 2605.6 | 2746.9 | 31.09 | 7.88 / 8.73 | |

R2VA 的耗时和内存消耗同时受输入视频时长和输出时长影响。表中采用等长输入和输出。
5、10、15 秒输入经过内部对齐，参考编码分别使用 107、226、345 帧。
FL2VA 样片先生成 124、243、362 个对齐帧，再保留首尾帧调整为 120、240、360 帧。
其中 362 帧通过本地样片脚本运行。

## 许可证

仓库原创代码和示例输入采用 [Apache-2.0](LICENSE)，再分发时保留[署名](NOTICE)。
上游模型、转换权重和模型输出适用各自条款，请遵守上游使用、地域及分发要求。
