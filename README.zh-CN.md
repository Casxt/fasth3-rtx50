# FastH3 RTX50

[English](README.md)

FastH3 RTX50 是面向内存和显存受限场景优化的 FastH3 项目，仅支持 NVIDIA RTX 50 系列显卡。
各组件按需组合使用 BF16 和 MXFP8，在兼顾速度和画质的前提下尽量降低内存和显存开销。
核心算子采用 CuTe DSL 实现，便于学习和参考。四种视频与立体声音频生成任务共用一个命令行入口，
模型权重在本地下载和转换。

目前的性能优化主要集中在 DiT，VAE 和 Qwen 条件编码器尚未深入调优。

主 DiT 层默认复用单个 GPU 权重 arena。长序列在显存预算允许时将 MXFP8 注意力输出保留在 GPU，
预估工作区无法容纳时自动使用主存暂存路径。

**建议使用编程 Agent 辅助安装、配置和执行。** 让 Agent 读取仓库内的
[AGENTS.md](AGENTS.md)，其中包含依赖、权重准备、提示词增强、测试和排错说明。
可以直接给 Agent 以下任务。

```text
先阅读 AGENTS.md，检查这台机器及已有的环境和模型文件，配置固定版本的依赖，
为我选择的任务准备权重，并运行 README 中的示例。核对输出，分别报告耗时、
Python 进程内存、全系统内存和显存占用。
```

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

`--task t2va` 和 `--task fl2va` 默认准备 FastH3 T2VA 文生视频权重。
`--task f2va` 和 `--task r2va` 准备 FastH3 Omni 权重。
用 `--task fl2va --model fasth3-omni-ref2va` 可单独准备首尾帧任务的备选权重。
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

FL2VA 默认使用 FastH3 T2VA 文生视频权重。
在 `fl2va` 命令中加 `--model fasth3-omni-ref2va`，即可使用 R2VA 同源的 Omni 权重，
输入仍是首帧和尾帧。两套权重在动作、细节和首尾过渡上的表现随场景而变，
建议用自己的关键帧和提示词对照测试。

F2VA 使用 Omni 并提供首帧条件。R2VA 使用 Omni，支持按顺序传入多个 `--reference`，接受 `image:PATH`、
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

使用 FastH3 T2VA 文生视频权重时，F2VA 和 FL2VA 可能出现轻微模糊。在一条使用该权重的
5 秒 F2VA 对照样片中，将 DiT 从 MXFP8 换成 BF16 后，模糊没有明显改善。我们的帧控样片
使用 R2VA 同源的 Omni 参考权重时避开了这类模糊。F2VA 已使用 Omni，FL2VA 可通过
`--model fasth3-omni-ref2va` 切换。

## 样片与实测性能

Windows、RTX 5080 16 GB，1344×768、24 fps，PyTorch 显存上限 10 GiB，整卡至少预留 256 MiB。
每行是一次实测。生成耗时含本地推理与导出，全流程另含 OpenH3-IR 与当次样片封装。
沿用已改写提示词的样片只列生成耗时（全流程列为 —）。
Python 内存是进程树工作集 / 私有提交，
全系统内存是物理占用（总量减可用量）/ 提交量，均为生成期间各自的采样峰值，单位为 GiB。
CUDA 两项是分配峰值和保留显存采样峰值。仓库样片统一为 H.265 CRF 20，转码耗时未计入。

| 模式 | 输入 | 输出时长 | 生成（秒） | 全流程（秒） | Python 内存（GiB） | 全系统内存（GiB） | CUDA 分配 / 保留（GiB） | 样片 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| T2VA | 仅文本 | 5 秒 | 230.8 | 270.2 | 8.78 / 22.74 | 27.57 / 48.57 | 3.65 / 4.52 | [MP4](examples/demos/t2va-5s.mp4) |
| T2VA | 仅文本 | 10 秒 | 460.3 | 496.0 | 9.07 / 23.06 | 27.96 / 48.94 | 3.85 / 6.10 | [MP4](examples/demos/t2va-10s.mp4) |
| T2VA | 仅文本 | 15 秒 | 720.2 | 745.0 | 9.47 / 23.52 | 28.38 / 49.92 | 5.15 / 7.65 | [MP4](examples/demos/t2va-15s.mp4) |
| F2VA | 1 张首帧图 | 5 秒 | 236.5 | 278.4 | 16.86 / 28.42 | 35.50 / 54.51 | 3.66 / 4.52 | [MP4](examples/demos/f2va-5s.mp4) |
| F2VA | 1 张首帧图 | 10 秒 | 414.3 | 448.7 | 16.86 / 28.81 | 36.00 / 55.18 | 3.70 / 5.78 | [MP4](examples/demos/f2va-10s.mp4) |
| F2VA | 1 张首帧图 | 15 秒 | 673.7 | 705.1 | 16.86 / 29.27 | 35.70 / 55.76 | 4.77 / 7.34 | [MP4](examples/demos/f2va-15s.mp4) |
| FL2VA (Omni) | 2 张图（首尾帧） | 5 秒 | 246.1 | 321.4 | 16.91 / 28.52 | 36.09 / 54.84 | 3.67 / 4.54 | [MP4](examples/demos/fl2va-5s.mp4) |
| FL2VA (Omni) | 2 张图（首尾帧） | 10 秒 | 456.3 | 518.6 | 16.95 / 28.89 | 36.13 / 55.03 | 3.71 / 5.92 | [MP4](examples/demos/fl2va-10s.mp4) |
| FL2VA (Omni) | 2 张图（首尾帧） | 15 秒 | 769.4 | 855.7 | 16.91 / 29.24 | 36.30 / 54.95 | 4.86 / 7.46 | [MP4](examples/demos/fl2va-15s.mp4) |
| FL2VA (T2VA) | 2 张图（首尾帧） | 5 秒 | 266.0 | — | 16.92 / 28.32 | 37.39 / 56.17 | 3.67 / 4.67 | [MP4](examples/demos/fl2va-t2va-5s.mp4) |
| FL2VA (T2VA) | 2 张图（首尾帧） | 10 秒 | 527.9 | — | 16.91 / 28.69 | 37.46 / 56.23 | 4.01 / 6.35 | [MP4](examples/demos/fl2va-t2va-10s.mp4) |
| FL2VA (T2VA) | 2 张图（首尾帧） | 15 秒 | 842.4 | — | 16.91 / 29.08 | 37.39 / 56.82 | 5.32 / 7.92 | [MP4](examples/demos/fl2va-t2va-15s.mp4) |
| R2VA | 1 张参考图 + 1 段 5 秒视频 | 5 秒 | 530.8 | 583.5 | 17.34 / 28.95 | 39.74 / 54.89 | 3.74 / 6.38 | [MP4](examples/demos/r2va-5s.mp4) |
| R2VA | 1 张参考图 + 1 段 10 秒视频 | 10 秒 | 1506.4 | 1669.0 | 17.81 / 30.33 | 33.52 / 50.84 | 5.57 / 9.39 | [MP4](examples/demos/r2va-10s.mp4) |
| R2VA | 1 张参考图 + 1 段 15 秒视频 | 15 秒 | 2280.7 | — | 18.19 / 31.19 | 33.86 / 51.84 | 8.17 / 9.88 | [MP4](examples/demos/r2va-15s.mp4) |

R2VA 的输入和输出时长都会影响耗时和内存。三条替换样片分别截取短剧的 0–5、5–15、15–30 秒，
每段输出与输入等长，均以同一张 Kimi 参考图替换蓝发角色。10 秒 R2VA 一行是基线实测，
15 秒一行使用了 DiT 缓冲复用。FL2VA 样片保留首尾帧，
发布版调整为 120、240、360 帧。15 秒样片使用本地脚本生成 362 个对齐帧。

**接续短剧** · [The Spare Moon（30 秒）](examples/demos/the-spare-moon-30s.mp4) · 首帧生成 15 秒后，
取其最后一帧作为下一段首帧，再用首尾帧生成 15 秒。
两段生成合计 1442.0 秒，全流程 1555.8 秒。Python 内存峰值 16.93 / 29.27 GiB，
全系统内存峰值 36.20 / 55.76 GiB，CUDA 分配 / 保留峰值 4.88 / 7.58 GiB。

### 480p 测速

两条样片均为 832×480、24 fps、5 秒，复用 768p 样片的已改写提示词。耗时含生成与导出。

| 模式 | 输入 | 生成（秒） | Python 内存（GiB） | 全系统内存（GiB） | CUDA 分配 / 保留（GiB） |
| --- | --- | ---: | ---: | ---: | ---: |
| T2VA | 仅文本 | 149.2 | 8.50 / 22.01 | 24.32 / 42.83 | 3.52 / 4.03 |
| R2VA | 1 张参考图 + 1 段 5 秒视频 | 186.8 | 17.08 / 27.90 | 32.95 / 48.71 | 3.56 / 4.09 |

## 许可证

仓库原创代码和示例输入采用 [Apache-2.0](LICENSE)，再分发时保留[署名](NOTICE)。
上游模型、转换权重和模型输出适用各自条款，请遵守上游使用、地域及分发要求。
