# 余音 · Karaoke

把 YouTube / Bilibili 视频变成可以唱的卡拉 OK：网页点歌、后台预处理、共享歌单和自动播放下一首。

## 当前功能

- 输入歌名／歌手搜索 YouTube、B站，预览后直接加入歌单；也支持视频链接。后台准备与播放顺序分别维护。
- 自动下载视频，分离背景音乐、伴唱、主唱，生成带字幕的 MV。
- 中文、日文和英文歌词对齐；当前句＋下一句，琥珀色逐字／词点亮，硬切换句，片头歌曲信息和四秒开唱提示。
- 暂停、定位、重唱、下一首、全屏、音量；三轨实时升降调，切换纯伴奏、保留伴唱和原唱。
- 伴奏、伴唱、主唱独立音量，支持全屏调整和遥控同步；默认 10 GiB 歌曲缓存上限与 LRU 自动回收。
- SQLite 持久化，SSE 推送进度；同一局域网可以用手机遥控点歌。

这是本地单房间 MVP。音频使用原曲和分离音轨，不生成替代音乐。当前有神经网络分离和对齐，OpenAI API 用于转录和歌曲信息，歌词按转录原文分行；尚未接入 Codex。切字精度仍在迭代，行为测试通过不代表音质或切字准确率已经达标。失败样本可先运行音频／字幕回归，通过后再补视频，见 [API](karaoke/API.md#先验证音频)。

## 先启动网页

需要 Python 3.12。在仓库根目录运行：

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r karaoke/requirements-api.txt
.venv/bin/python -m karaoke.api
```

打开 <http://127.0.0.1:8746/>，接口文档在 <http://127.0.0.1:8746/docs>。前端不需要构建。

**仓库不包含歌曲数据，初次启动曲库为空。** 上面的步骤可以启动界面和队列；真正处理新歌还需要 [安装说明](SETUP.md) 中的 FFmpeg、LAME、推理依赖、模型权重和 `OPENAI_API_KEY`。

## 文档与指令

- [开发指令](AGENTS.md)：给 Codex 和后续开发者的项目约束。
- [安装与模型准备](SETUP.md)：环境、固定版本的上游源码、权重位置、验证命令。
- [网页使用说明](karaoke/APP.md)：点歌、播放、混音模式和遥控。
- [API](karaoke/API.md)：请求、任务状态、成品下载和环境变量。
- [界面设计](karaoke/docs/design/app.md) · [字幕设计](karaoke/docs/design/subtitles.md)。
- [切字验收](karaoke/docs/word-onset-metrics.md) · [主唱／伴唱验收](karaoke/docs/lead-backing-metrics.md) · [整体验收](karaoke/ACCEPTANCE.md)。
- [自动质量检查](karaoke/docs/automatic-quality.md)：独立参考评分、疑点检查器的漏报／误报和故障注入验证。

## 代码结构

```text
karaoke/
  api.py, room.py, job_schema.py    HTTP、任务队列、共享播放状态
  pipeline.py, song_sources.py     下载、分离、歌词、对齐、导出
  web/                            原生 HTML / CSS / JavaScript 播放器
  exact_stems.py, three_stems.py   可重组的双轨、三轨母版
  align_*.py, phoneme_progress.py  词／音素对齐与保护规则
  subtitle*.py, render_*.py        字幕布局、点亮、视频导出
  benchmark_*.py, dsp.py           算法实验与评测工具
  experiments/cjk-onsets-v003/    当前流程仍使用的多窗口对齐代码
  dependencies/                  权重来源与校验值（不含权重）
  models/pretrained/*.yaml        分离模型配置（不含权重）
```

本仓库上传当前代码、测试、配置和说明；不上传歌曲、歌词库、音轨、MV、封面、任务数据库、实验产物、模型权重、大型第三方研究仓库或虚拟环境。浏览器升降调使用的 114 KB Signalsmith Stretch 运行库随源码提供，保留 MIT 许可和固定版本来源。历史报告中提到的试听页、实验数据和本机路径不随仓库发布；旧实验脚本需要自行准备相应数据。

## 验证

准备完整 Python 环境并按 [安装说明](SETUP.md) 拉取 SOFA 源码后：

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=karaoke .venv/bin/python -m pytest -q karaoke/test_api.py karaoke/test_room.py karaoke/test_algorithms.py
node --test karaoke/test_player.mjs karaoke/test_pitch.mjs karaoke/test_search.mjs
```

这些测试使用临时文件和合成音频，无需歌曲库、模型权重或 API key。中文发音测试会读取 SOFA 的音素字典，日文发音依赖 pyopenjtalk 字典。
