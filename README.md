# 退学 KTV · Karaoke

把有权下载和处理的 YouTube 视频变成可以唱的卡拉 OK：网页点歌、后台预处理、共享歌单和自动播放下一首。

## 界面预览

以下为当前界面的实际截图，展示示例曲库、搜索和字幕播放。图中歌曲、封面、视频画面及歌词属于第三方，不包含在本项目的 MIT 授权中；素材出处与授权状态见[截图说明](docs/images/README.md)。截图用于说明功能，不代表逐字起音准确率已经通过人工验收。

### 全屏播放与字幕

琥珀色逐字／词点亮，下一句提前显示。粤语歌词带粤语拼音（粤拼），日文歌词的汉字带平假名注音，读音显示在对应文字上方；底部保留播放进度、主唱音量和升降调。

![全屏播放：粤拼注音、琥珀色点亮、下一句预读与底部控制栏](docs/images/playback.png)

### 桌面点歌台

按歌手浏览、从曲库点歌，右侧安排待唱顺序。

![桌面点歌台：歌手、曲库、待唱列表与底部播放控制栏](docs/images/desktop.png)

### 综合搜索

搜索歌名，分别展示曲库里的歌与在线结果，保留不同歌手和版本。

![综合搜索：Lemon 的曲库匹配项与在线搜索结果](docs/images/search-general.png)

### 歌手搜索

搜索歌手后直接展开作品列表，可切换其他候选并逐首点歌。

![歌手搜索：陈奕迅的信息、曲库匹配项与作品列表](docs/images/search-artist.png)

### 正在播放

普通播放页同时显示字幕画面、待唱列表和播放控制。

![正在播放：字幕 MV、右侧待唱列表与底部播放控制栏](docs/images/now-playing.png)

### 手机待唱 · 夜间模式

待唱顺序和播放控制同屏显示。

<img src="docs/images/mobile.png" alt="手机夜间模式：待唱列表与底部播放控制栏" width="320">

## 当前功能

- 按歌名查 YouTube Music 歌曲，或找歌手浏览热门歌曲；下拉自动加载更多，支持播放量排序、预览和直接点歌，也保留原始 YouTube 视频搜索和视频链接。后台准备与播放顺序分别维护。
- 自动下载视频，分离背景音乐、伴唱、主唱，生成带字幕的 MV。
- 多语言歌词对齐（中日英保留专用唱歌模型细化）；当前句＋下一句，琥珀色逐字／词点亮，硬切换句，片头歌曲信息和四秒开唱提示。
- 粤语拼音与日文平假名：粤语歌词显示粤拼，日文歌词的汉字显示平假名注音，在生成字幕时完成。
- 暂停、定位、重唱、下一首、全屏；混音后实时升降调，主唱音量按百分比调节。
- 伴奏和伴唱始终开启，主唱 0–100% 加减按钮支持全屏调整，每次 5 个百分点；各设备独立播放；默认 10 GiB 歌曲缓存上限与 LRU 自动回收。
- SQLite 持久化，SSE 推送进度；同一局域网可以用手机点歌和播放。

这是本地单房间 MVP。音频使用原曲和分离音轨，不生成替代音乐。当前有神经网络分离和对齐，普通点歌优先获取已发表歌词，模型负责歌曲信息与声学对齐；无可靠文本时保留音轨，不用识别草稿充当歌词。独立音频验收仍保留 ASR 覆盖率检查。切字精度仍在迭代，行为测试通过不代表音质或切字准确率已经达标。失败样本可先运行音频／字幕回归，通过后再补视频，见 [API](karaoke/API.md#先验证音频)。

## 先启动网页

需要 Python 3.12。在仓库根目录运行：

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r karaoke/requirements-api.txt
.venv/bin/python -m karaoke.api
```

打开 <http://127.0.0.1:8746/>，接口文档在 <http://127.0.0.1:8746/docs>。前端不需要构建。

**仓库不包含歌曲数据，初次启动曲库为空。** 上面的步骤可以启动界面和队列；真正处理新歌还需要 [安装说明](SETUP.md) 中的 FFmpeg、Node.js、推理依赖和模型权重。在线识别／元数据整理需要 `OPENAI_API_KEY`；也可配置本地识别模式。

## 文档与指令

- [贡献指南](CONTRIBUTING.md)：开发、测试和安全提交。
- [开发指令](AGENTS.md)：给 Codex 和后续开发者的项目约束。
- [安装与模型准备](SETUP.md)：环境、固定版本的上游源码、权重位置、验证命令。
- [网页使用说明](karaoke/APP.md)：点歌、播放、主唱音量和多设备播放。
- [API](karaoke/API.md)：请求、任务状态、成品下载和环境变量。
- [界面设计](karaoke/docs/design/app.md) · [字幕设计](karaoke/docs/design/subtitles.md)。
- [切字验收](karaoke/docs/word-onset-metrics.md) · [主唱／伴唱验收](karaoke/docs/lead-backing-metrics.md) · [整体验收](karaoke/ACCEPTANCE.md)。
- [自动质量检查](karaoke/docs/automatic-quality.md)：独立参考评分、疑点检查器的漏报／误报和故障注入验证。

## 代码结构

```text
karaoke/
  api.py, room.py, job_schema.py    HTTP、任务队列、共享歌单和准备状态
  pipeline.py, song_sources.py     下载、分离、歌词、对齐、导出
  web/                            原生 HTML / CSS / JavaScript 播放器
  exact_stems.py, three_stems.py   可重组的双轨、三轨母版
  align_*.py, phoneme_progress.py  词／音素对齐与保护规则
  subtitle*.py, render_*.py        字幕布局、点亮、视频导出
  benchmark_neural.py             正式流程使用的分离子进程入口
  benchmark_*.py, dsp.py           其余算法实验与评测工具
  experiments/cjk-onsets-v003/    当前流程仍使用的多窗口对齐代码
  dependencies/                  权重来源与校验值（不含权重）
  models/pretrained/*.yaml        分离模型配置（不含权重）
```

详细入口见[源码导航](karaoke/README.md)。仓库保存代码、测试、配置、文档与上述界面截图；运行所需的歌曲、歌词库、音轨、MV、任务数据库、实验产物、模型权重、大型第三方研究仓库和虚拟环境由使用者另行准备。

## 许可与部署边界

项目自行编写的代码与文档采用 [MIT License](LICENSE)。截图内的第三方素材，以及第三方代码、配置、数据和模型，保留各自权利与许可，见 [THIRD_PARTY.md](THIRD_PARTY.md)。其中包含 GPL 组件和可选的非商业模型；MIT 不替代这些条款，也不代表完整运行环境可以仅按 MIT 分发。

软件许可不授予歌曲、录音、视频、封面或歌词的使用权。下载、处理及分享前，请确认相应内容权限并遵守来源平台条款；开源或非商业使用不自动免除这些要求。

服务默认只监听本机，没有内置账号隔离。远程访问需由部署者保护整个站点及 API；媒体、SSE 和操作接口都在保护范围内。歌曲、歌词和个人凭据不应加入源码发布包。

## 验证

[GitHub Actions](https://github.com/Trinkle23897/karaoke/actions/workflows/tests.yml) 在每次 push 和 PR 时运行全部 Python 与 JavaScript 行为测试，并检查空数据目录启动。测试报告保留 14 天，也可从 Actions 页面手动运行。

准备完整 Python 环境并按 [安装说明](SETUP.md) 拉取 SOFA 源码后：

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=karaoke .venv/bin/python -m pytest -q karaoke/test_api.py karaoke/test_room.py karaoke/test_algorithms.py
node --test karaoke/test_player.mjs karaoke/test_pitch.mjs karaoke/test_search.mjs
```

这些测试使用临时文件和合成音频，无需歌曲库、模型权重或 API key。中文发音测试会读取 SOFA 的音素字典，日文发音依赖 pyopenjtalk 字典。
