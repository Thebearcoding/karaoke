# 安装与运行

所有命令从仓库根目录执行。以下保留目前实际使用的目录结构和版本；第三方源码、权重及生成数据都由 `.gitignore` 排除。

## 1. 网页与任务服务

只启动界面不需要下载模型：

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r karaoke/requirements-api.txt
.venv/bin/python -m karaoke.api
```

地址：<http://127.0.0.1:8746/>。初次启动是空曲库。服务会创建 `karaoke/service-data/`；它不是源代码，不要提交。

## 2. 完整处理环境

需要 `ffmpeg`、`ffprobe`、`lame` 在 PATH 中，例如 macOS 使用 `brew install ffmpeg lame`，Debian/Ubuntu 使用 `sudo apt-get install ffmpeg lame libsndfile1`。LAME 命令行负责 MP3 试听文件的完整尾部填充信息；FFmpeg 负责母版解码、公共增益转换与视频编码。Node.js 仅用于运行播放器和音频变调测试，前端本身不需要构建。实时升降调的 Signalsmith Stretch / WASM 随源码提供，不需要额外安装、模型下载或运行时 CDN；版本和许可见 `karaoke/dependencies/pitch-shift.json`。

主环境负责音频导出、ASR、歌词整理、对齐及字幕渲染。先按硬件安装匹配版本的 PyTorch 与 torchaudio，再装项目依赖。当前 Mac / MPS 开发环境使用 Python 3.12、`torch==2.11.0`、`torchaudio==2.11.0`：

```sh
.venv/bin/python -m pip install torch==2.11.0 torchaudio==2.11.0
.venv/bin/python -m pip install -r requirements-dev.txt
```

CUDA 机器应选择与驱动兼容的 PyTorch CUDA wheel，保持 torch/torchaudio 版本匹配。CPU 可以执行，但整曲处理会慢。这里记录的是实际用过的主依赖，不是跨平台完整 lockfile；CUDA 4090 尚未完成端到端验证。

分离使用独立环境，避免 audio-separator 的依赖影响主环境：

```sh
python3.12 -m venv .venv-separator
.venv-separator/bin/python -m pip install 'audio-separator==0.44.5'
export KARAOKE_SEPARATOR_PYTHON="$PWD/.venv-separator/bin/python"
```

该环境也需安装适合硬件的 PyTorch。当前分离开发环境使用 torch 2.13.0。`benchmark_neural.py` 会检查 audio-separator 的版本，并显式设置模型类型、重叠和推理配置；不要只换一份名字相似的权重。

## 3. 对齐器源码

### 中文／日文：SOFA

```sh
mkdir -p karaoke/research/subtitles-2026-09-18
git clone --filter=blob:none --no-checkout https://github.com/qiuqiao/SOFA.git karaoke/research/subtitles-2026-09-18/sofa
git -C karaoke/research/subtitles-2026-09-18/sofa checkout --detach 9549c6a86d16019c817eefe4bb8183405da524cc
```

中文发音测试也需要该源码中的 `dictionary/opencpop-extension.txt`，但不需要模型权重。日文使用 pyopenjtalk；首次调用可能需要下载它自己的发音字典。不要直接覆盖安装 SOFA 的旧版完整 requirements；本项目主环境使用 `requirements-full.txt` 中的依赖。

### 英文：LyricsAlignment-Multilingual

```sh
mkdir -p karaoke/research/word-onset-2026-09-18
git clone --filter=blob:none --no-checkout https://github.com/jhuang448/LyricsAlignment-Multilingual.git karaoke/research/word-onset-2026-09-18/lyrics-alignment-multilingual
git -C karaoke/research/word-onset-2026-09-18/lyrics-alignment-multilingual checkout --detach ca1a3923d6c8bf7d20eefa1080f3f104a6173d38
cp karaoke/dependencies/english-alignment.json karaoke/research/word-onset-2026-09-18/lyrics-alignment-multilingual/source-manifest.json
```

需要其中的 `model.py`、`utils.py` 和 `checkpoints/checkpoint_Baseline`。源码和 checkpoint 的 SHA-256 见 [english-alignment.json](karaoke/dependencies/english-alignment.json)。该上游 checkout 含 checkpoint；整个第三方目录在 Git 排除范围内。不要安装它的旧版训练 requirements 覆盖主环境。

## 4. 模型权重

模型文件不在仓库里。下载后放在下列位置，保留原始文件，不转换、不重命名为另一模型。仓库中的 YAML 是与当前流程配套的配置。

| 用途 | 下载来源 | 本地位置 |
|---|---|---|
| F：音乐／人声分离 | [Mel-Band RoFormer](https://huggingface.co/KimberleyJSN/melbandroformer/resolve/main/MelBandRoformer.ckpt) | `karaoke/models/pretrained/vocals_mel_band_roformer.ckpt` |
| 主唱／伴唱分离 | [Anvuew 固定版本](https://huggingface.co/anvuew/karaoke_bs_roformer/resolve/0d4423d42e12cf2ba39ae09171028507b8a2a7be/karaoke_bs_roformer_anvuew.ckpt) | `karaoke/models/pretrained/karaoke_bs_roformer_anvuew.ckpt` |
| 中文音素对齐 | [SOFA Mandarin](https://github.com/qiuqiao/SOFA/releases/download/v1.0.1/pretrained_mandarin_singing.zip) | 解压到 `karaoke/models/alignment/sofa/mandarin/`，保留压缩包子目录 |
| 日文音素对齐 | [SOFA Japanese](https://github.com/ariikamusic/SOFA_Models/releases/download/akm_ja_v001/akm_ja_v001.zip) | 解压到 `karaoke/models/alignment/sofa/japanese/`，保留压缩包子目录 |
| 英文音素对齐 | 上一步的固定版本仓库 | `karaoke/research/word-onset-2026-09-18/lyrics-alignment-multilingual/checkpoints/checkpoint_Baseline` |
| Whisper 粗对齐 | stable-whisper 自动下载 | `karaoke/models/alignment/whisper/` |

权重和压缩包的校验值分别记录在 [separation.json](karaoke/dependencies/separation.json)、[sofa-mandarin.json](karaoke/dependencies/sofa-mandarin.json)、[sofa-japanese.json](karaoke/dependencies/sofa-japanese.json)。`separation.json` 也保留了早期对照模型的来源；正式流程只需要 F 和 Anvuew。可用 `shasum -a 256 文件路径` 核对下载内容。上游若更新了 `main` 对应的文件，应以这里记录的 SHA-256 判断是否还是同一个模型。

这些模型和上游源码遵循各自的许可。仓库只保存调用代码、配置和来源，不重新分发其权重或第三方源码。

## 5. 运行配置

通过环境注入 `OPENAI_API_KEY`，不要写进源代码、命令示例或提交 `.env`。当前在线调用用于 ASR 和歌词／元数据整理，文本模型默认 `gpt-5-mini`。

```sh
export KARAOKE_SEPARATOR_PYTHON="$PWD/.venv-separator/bin/python"
export KARAOKE_DEVICE=auto
.venv/bin/python -m karaoke.api
```

`auto` 按 CUDA → MPS → CPU 选择设备。Linux 还需用 `KARAOKE_FONT_FILE` 指定覆盖目标语言的字体文件，例如已安装的 Noto Serif CJK 字体。程序会检查缺字；不要用缺字方块继续导出。完整环境变量见 [API.md](karaoke/API.md)。

保持一个 Uvicorn 进程、一个处理 worker；不要对同一数据目录使用多 worker。需要局域网遥控时再显式设置 `KARAOKE_HOST=0.0.0.0`，手机访问 `http://电脑IP:8746/?remote=1`。

长期在本机运行时，将服务与临时终端分开，并保存日志（macOS / Linux）：

```sh
mkdir -p karaoke/service-data
nohup .venv/bin/python -u -m karaoke.api </dev/null >>karaoke/service-data/server.log 2>&1 &
```

环境变量需在启动前设置。检查 `curl -fsS http://127.0.0.1:8746/health` 和 `tail -n 50 karaoke/service-data/server.log`；启动终端退出后应再次检查健康接口。`nohup` 不负责崩溃重启或开机启动；需要这两项时交给操作系统的服务管理器。重启前确认没有正在准备或播放的歌曲，再向监听进程发送 `TERM`，等它退出后启动，避免两个 worker 抢同一数据目录。

网页能打开只说明本地服务正常，不代表下载网络可用。若日志出现 `Unable to connect to proxy` / `Tunnel connection failed: 403 Forbidden`，检查启动进程继承的代理环境。通过 Codex 启动长期服务时，应使用已获准访问视频站点的运行环境；不要把工具临时注入的受限代理当作应用代理，也不要在应用代码里擅自删除用户的代理设置。可在同一运行环境先执行 `python -m yt_dlp --ignore-config --skip-download --print title 视频URL` 验证。

## 6. 测试与范围

安装开发依赖、主环境 PyTorch 和 SOFA 源码后：

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=karaoke .venv/bin/python -m pytest -q karaoke/test_api.py karaoke/test_room.py karaoke/test_algorithms.py
node --test karaoke/test_player.mjs karaoke/test_pitch.mjs karaoke/test_search.mjs
```

这些行为测试不发起在线模型调用，不要求歌曲库或模型权重。真正验证下载、分离、对齐和渲染，仍需完成以上资源准备并提交一个真实视频链接。首次模型／字典下载耗时与后续单曲处理耗时应分别记录。

`benchmark_*.py`、OCR 对照脚本及历史实验说明属于研究工具；它们可能引用本机实验素材或可选模型，不是空仓库一键可运行的演示。正式入口是 `python -m karaoke.api`。
