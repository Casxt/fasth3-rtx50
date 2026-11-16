# FastH3 RTX50

[English](README.md)

FastH3 RTX50 是面向家用 NVIDIA RTX 50 系列显卡的高性能 FastH3 视频与立体声音频生成实现，
针对内存和显存受限场景优化，仅支持 RTX 50 系列显卡。
各组件按需组合使用 BF16 和 MXFP8，在兼顾速度和画质的前提下尽量降低内存和显存开销。
MXFP8 计算为每组 32 个元素使用独立缩放因子，更细致地适配局部数值范围，控制量化误差，
尽可能保留权重和激活的数值精度。
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

Windows x64 锁定 Python 3.12.13 与 CUDA 12.8 版 PyTorch。
两个平台的启动脚本共用 `scripts/bootstrap.py`。已有 Python 时可直接执行
`python scripts/bootstrap.py`；Windows 还需在 `PATH` 中提供 uv。
给这条 Python 命令加 `--dry-run` 可预览安装计划。

## Linux / WSL 安装

Linux x86_64 环境已在 Ubuntu 22.04、Python 3.12.3、PyTorch 2.10.0+cu128、
torchvision 0.25.0+cu128 和 CuTe DSL 4.8.1 上检查。使用 Python 3.12 和兼容
CUDA 12.8 的 NVIDIA 驱动。选择有足够空间存放环境、缓存和模型的工作目录；
WSL 文件保存在其 Linux 文件系统中。在仓库根目录替换示例路径后执行：

```bash
export FASTH3_WORK_DIR=/path/to/data-disk/fasth3
export XDG_CACHE_HOME="$FASTH3_WORK_DIR/cache" TMPDIR="$FASTH3_WORK_DIR/tmp"
export CUDA_CACHE_PATH="$FASTH3_WORK_DIR/cache/cuda" TRITON_CACHE_DIR="$FASTH3_WORK_DIR/cache/triton"
export CUTE_DSL_CACHE_DIR="$FASTH3_WORK_DIR/cache/cutlass" CUTE_DSL_DUMP_DIR="$FASTH3_WORK_DIR/tmp/cute-dump"
python3.12 scripts/bootstrap.py
source "$FASTH3_WORK_DIR/.venv-linux/bin/activate"
fasth3 --help
fasth3 doctor --gpu
```

显卡空闲时运行 GPU 检查。脚本创建独立 venv，先安装 `requirements-linux-cu128.txt`
中的 CUDA 栈，再安装 `requirements-linux.txt` 中的固定版本和本项目，包含 `pe,test`
依赖并检查包兼容性。可用 `FASTH3_PYTHON` 选择其他 Python 3.12 可执行文件，
用 `FASTH3_PYPI_INDEX` 选择 PyPI 镜像。`bash scripts/bootstrap-linux.sh` 只是同一
Python 入口的薄启动脚本。后续运行的 shell 也需保留上述缓存设置；
脚本自身的 export 不会改变父 shell。后文准备权重和生成示例中的模型目录应改为
`--models-dir "$FASTH3_WORK_DIR/models"`，将权重也保存在所选磁盘；工作目录设置
不会改变 CLI 的模型路径。

Linux 使用 cuda-python/cuda-bindings 12.9.4，以满足 Linux PyTorch wheel 对 binding
版本的严格要求；Windows 保持 cuda-python 12.9.10。Linux 依赖文件是版本快照，
仅固定的 Diffusers 源码包带哈希，并非完整 wheel 哈希锁。共享 bootstrap 已在复用
现有依赖缓存的条件下创建并安装新的 `.venv-linux`，耗时 16.789 秒；连同包兼容性检查、
CPU/GPU doctor 和 3 项真实 CuTe 内核检查，验证总耗时 36.297 秒。完整 CPU 套件为
403 通过 / 51 跳过，耗时 4.69 秒；完整 GPU 套件为 454 通过 / 零跳过，耗时 39.64 秒，
包含 10 GiB allocator 上限下的真实 NVFP4 checkpoint 检查。
资源统计和检查细节见 [AGENTS.md](AGENTS.md)。

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

实测中 480p 输出的画质较差，主要用于测速。建议使用 720p 及以上，优先采用默认 768p 画布。

T2VA 有时会出现块状模糊，在脸和头发上较明显。程度随 seed 和单次生成而变化，
需要挑片。

使用 FastH3 T2VA 文生视频权重时，F2VA 和 FL2VA 可能出现轻微模糊。在一条使用该权重的
5 秒 F2VA 对照样片中，将 DiT 从 MXFP8 换成 BF16 后，模糊没有明显改善。我们的帧控样片
使用 R2VA 同源的 Omni 参考权重时避开了这类模糊。F2VA 已使用 Omni，FL2VA 可通过
`--model fasth3-omni-ref2va` 切换。

## 样片与实测性能

角色署名及独立于代码许可证的素材使用条件，见[样片来源与使用条件](examples/demos/README.md)。

**T2VA · 5 秒（Linux RTX 5090）**

https://github.com/user-attachments/assets/92b28005-e91a-4202-8e2d-9a81cd898797

**R2VA · 10 秒（Windows RTX 5080）**

https://github.com/user-attachments/assets/dc83d8f6-61ae-4fb6-b4cc-a4e09a57ebe1

表中样片使用 raw MP4 直链。若 GitHub 未在新窗口打开，请按住 Ctrl / ⌘ 点击链接。

Windows、RTX 5080 16 GB，1344×768、24 fps，PyTorch 显存上限 10 GiB，整卡至少预留 256 MiB。
每行是一次实测。生成耗时含本地推理与导出，全流程另含 OpenH3-IR 与当次样片封装。
沿用已改写提示词的样片只列生成耗时（全流程列为 —）。
Python 内存是进程树工作集 / 私有提交，
全系统内存是物理占用（总量减可用量）/ 提交量，均为生成期间各自的采样峰值，单位为 GiB。
CUDA 两项是分配峰值和保留显存采样峰值。仓库样片统一为 H.265 CRF 20，转码耗时未计入。

| 模式 | 输入 | 输出时长 | 生成（秒） | 全流程（秒） | Python 内存（GiB） | 全系统内存（GiB） | CUDA 分配 / 保留（GiB） | 样片 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| T2VA | 仅文本 | 5 秒 | 230.8 | 270.2 | 8.78 / 22.74 | 27.57 / 48.57 | 3.65 / 4.52 | <a href="https://raw.githubusercontent.com/Casxt/fasth3-rtx50/refs/heads/main/examples/demos/t2va-5s.mp4" target="_blank" rel="noopener noreferrer">MP4 ↗</a> |
| T2VA | 仅文本 | 10 秒 | 460.3 | 496.0 | 9.07 / 23.06 | 27.96 / 48.94 | 3.85 / 6.10 | <a href="https://raw.githubusercontent.com/Casxt/fasth3-rtx50/refs/heads/main/examples/demos/t2va-10s.mp4" target="_blank" rel="noopener noreferrer">MP4 ↗</a> |
| T2VA | 仅文本 | 15 秒 | 720.2 | 745.0 | 9.47 / 23.52 | 28.38 / 49.92 | 5.15 / 7.65 | <a href="https://raw.githubusercontent.com/Casxt/fasth3-rtx50/refs/heads/main/examples/demos/t2va-15s.mp4" target="_blank" rel="noopener noreferrer">MP4 ↗</a> |
| F2VA | 1 张首帧图 | 5 秒 | 236.5 | 278.4 | 16.86 / 28.42 | 35.50 / 54.51 | 3.66 / 4.52 | <a href="https://raw.githubusercontent.com/Casxt/fasth3-rtx50/refs/heads/main/examples/demos/f2va-5s.mp4" target="_blank" rel="noopener noreferrer">MP4 ↗</a> |
| F2VA | 1 张首帧图 | 10 秒 | 414.3 | 448.7 | 16.86 / 28.81 | 36.00 / 55.18 | 3.70 / 5.78 | <a href="https://raw.githubusercontent.com/Casxt/fasth3-rtx50/refs/heads/main/examples/demos/f2va-10s.mp4" target="_blank" rel="noopener noreferrer">MP4 ↗</a> |
| F2VA | 1 张首帧图 | 15 秒 | 673.7 | 705.1 | 16.86 / 29.27 | 35.70 / 55.76 | 4.77 / 7.34 | <a href="https://raw.githubusercontent.com/Casxt/fasth3-rtx50/refs/heads/main/examples/demos/f2va-15s.mp4" target="_blank" rel="noopener noreferrer">MP4 ↗</a> |
| FL2VA (Omni) | 2 张图（首尾帧） | 5 秒 | 246.1 | 321.4 | 16.91 / 28.52 | 36.09 / 54.84 | 3.67 / 4.54 | <a href="https://raw.githubusercontent.com/Casxt/fasth3-rtx50/refs/heads/main/examples/demos/fl2va-5s.mp4" target="_blank" rel="noopener noreferrer">MP4 ↗</a> |
| FL2VA (Omni) | 2 张图（首尾帧） | 10 秒 | 456.3 | 518.6 | 16.95 / 28.89 | 36.13 / 55.03 | 3.71 / 5.92 | <a href="https://raw.githubusercontent.com/Casxt/fasth3-rtx50/refs/heads/main/examples/demos/fl2va-10s.mp4" target="_blank" rel="noopener noreferrer">MP4 ↗</a> |
| FL2VA (Omni) | 2 张图（首尾帧） | 15 秒 | 769.4 | 855.7 | 16.91 / 29.24 | 36.30 / 54.95 | 4.86 / 7.46 | <a href="https://raw.githubusercontent.com/Casxt/fasth3-rtx50/refs/heads/main/examples/demos/fl2va-15s.mp4" target="_blank" rel="noopener noreferrer">MP4 ↗</a> |
| FL2VA (T2VA) | 2 张图（首尾帧） | 5 秒 | 266.0 | — | 16.92 / 28.32 | 37.39 / 56.17 | 3.67 / 4.67 | <a href="https://raw.githubusercontent.com/Casxt/fasth3-rtx50/refs/heads/main/examples/demos/fl2va-t2va-5s.mp4" target="_blank" rel="noopener noreferrer">MP4 ↗</a> |
| FL2VA (T2VA) | 2 张图（首尾帧） | 10 秒 | 527.9 | — | 16.91 / 28.69 | 37.46 / 56.23 | 4.01 / 6.35 | <a href="https://raw.githubusercontent.com/Casxt/fasth3-rtx50/refs/heads/main/examples/demos/fl2va-t2va-10s.mp4" target="_blank" rel="noopener noreferrer">MP4 ↗</a> |
| FL2VA (T2VA) | 2 张图（首尾帧） | 15 秒 | 842.4 | — | 16.91 / 29.08 | 37.39 / 56.82 | 5.32 / 7.92 | <a href="https://raw.githubusercontent.com/Casxt/fasth3-rtx50/refs/heads/main/examples/demos/fl2va-t2va-15s.mp4" target="_blank" rel="noopener noreferrer">MP4 ↗</a> |
| R2VA | 1 张参考图 + 1 段 5 秒视频 | 5 秒 | 496.3 | — | 17.41 / 28.92 | 34.93 / 53.26 | 3.73 / 6.38 | <a href="https://raw.githubusercontent.com/Casxt/fasth3-rtx50/refs/heads/main/examples/demos/r2va-5s.mp4" target="_blank" rel="noopener noreferrer">MP4 ↗</a> |
| R2VA | 1 张参考图 + 1 段 10 秒视频 | 10 秒 | 1344.3 | — | 17.90 / 29.80 | 36.18 / 54.10 | 6.64 / 7.88 | <a href="https://raw.githubusercontent.com/Casxt/fasth3-rtx50/refs/heads/main/examples/demos/r2va-10s.mp4" target="_blank" rel="noopener noreferrer">MP4 ↗</a> |
| R2VA | 1 张参考图 + 1 段 15 秒视频 | 15 秒 | 2258.0 | — | 18.34 / 30.64 | 35.76 / 54.76 | 8.49 / 8.74 | <a href="https://raw.githubusercontent.com/Casxt/fasth3-rtx50/refs/heads/main/examples/demos/r2va-15s.mp4" target="_blank" rel="noopener noreferrer">MP4 ↗</a> |

R2VA 的输入和输出时长都会影响耗时和内存。三项替换任务分别截取短剧的 0–5、5–15、15–30 秒，
每段输出与输入等长，均以同一张短边调整为 512 的 Kimi 参考图替换蓝发角色。
三条 768p R2VA 数据已使用 v0.1.1 默认配置、完整八步重新实测，并复用已改写提示词。
生成耗时包含推理、VAE 解码与导出，排除 PE 和 H.265 CRF 20 转码。
其他模态和 480p 结果沿用历史数据，v0.1.1 未重测。
本次 R2VA 直接使用生产 CLI：请求导出 120 / 240 / 360 帧，内部自动对齐至
124 / 243 / 362 帧后按请求帧数导出。
5 / 10 / 15 秒 R2VA 的 NVML 整卡显存采样峰值分别为 9.41 / 10.69 / 11.35 GiB，
包含驱动和其他进程占用，与表中 CUDA 分配 / 保留的统计口径不同。
历史 FL2VA 样片保留首尾帧，发布版调整为 120、240、360 帧。
历史 15 秒 FL2VA 样片使用本地脚本生成 362 个对齐帧后再调整导出。

**接续短剧** · <a href="https://raw.githubusercontent.com/Casxt/fasth3-rtx50/refs/heads/main/examples/demos/the-spare-moon-30s.mp4" target="_blank" rel="noopener noreferrer">The Spare Moon（30 秒） ↗</a> · 首帧生成 15 秒后，
取其最后一帧作为下一段首帧，再用首尾帧生成 15 秒。
两段生成合计 1442.0 秒，全流程 1555.8 秒。Python 内存峰值 16.93 / 29.27 GiB，
全系统内存峰值 36.20 / 55.76 GiB，CUDA 分配 / 保留峰值 4.88 / 7.58 GiB。

### 480p 测速

两条样片均为 832×480、24 fps、5 秒，复用 768p 样片的已改写提示词。耗时含生成与导出。

| 模式 | 输入 | 生成（秒） | Python 内存（GiB） | 全系统内存（GiB） | CUDA 分配 / 保留（GiB） |
| --- | --- | ---: | ---: | ---: | ---: |
| T2VA | 仅文本 | 149.2 | 8.50 / 22.01 | 24.32 / 42.83 | 3.52 / 4.03 |
| R2VA | 1 张参考图 + 1 段 5 秒视频 | 186.8 | 17.08 / 27.90 | 32.95 / 48.71 | 3.56 / 4.09 |

### Linux 验证

Linux RTX 5090 上的 17 项核心生成案例均完成了完整 8 次 transformer forward，
另完成一条 30 秒接续短片和橙球案例。对齐条件的 5 / 10 / 15 秒 R2VA 使用等长视频
加一张短边为 512 的参考图，输出为 1344×768、24 fps，耗时分别为
286.746 / 739.713 / 1260.393 秒；旧 Windows RTX 5080 记录为
496.331 / 1344.252 / 2258.036 秒，速度约为旧记录的 1.73–1.82 倍。
这是整机对照，CPU、操作系统和 I/O 条件有所不同。

完整案例与输入条件见[验证报告](benchmarks/linux-validation.md)，
硬件与传输实测见[带宽报告](benchmarks/transfer-baseline.md)。

## 许可证

仓库原创代码及通用橙球示例输入
[`examples/first.png`](examples/first.png)、[`examples/last.png`](examples/last.png)
采用 [Apache-2.0](LICENSE)，再分发时保留[署名](NOTICE)。
角色设计和使用这些角色的样片不在该代码许可范围内，见[样片来源与使用条件](examples/demos/README.md)。
上游模型、转换权重和模型输出仍适用各自条款，包括使用、地域及分发要求。
