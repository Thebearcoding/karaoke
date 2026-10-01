# 视频链接 → 卡拉 OK 成品

本地异步 API，复用当前 F + Anvuew 三轨分离、音频歌词对齐和金色字幕流程。

## 启动

在仓库根目录运行：

```sh
.venv/bin/python -m pip install -r karaoke/requirements-api.txt
.venv/bin/python -m karaoke.api
```

默认监听 `127.0.0.1:8746`。网页点歌台：<http://127.0.0.1:8746/>。交互文档：<http://127.0.0.1:8746/docs>。
API 的依赖文件只安装服务依赖。新检出请按[安装说明](../SETUP.md)准备模型、音频环境和 FFmpeg；模型权重和歌曲数据不在仓库中。

## 网页与共享歌单

网页使用 `GET /v1/room/events` 的 SSE 长连接接收完整状态快照，不轮询。断线后自动重连，重新获取当前状态。下面的 `GET /v1/jobs/{id}` 仍供脚本按需查询。

| 接口 | 用途 |
|---|---|
| `GET /v1/library` | 列出已知歌曲，实际文件完整时才标记为已准备 |
| `GET /v1/room` | 当前歌曲、播放位置和已点顺序 |
| `POST /v1/room/entries` | 传入 `{ "url": "..." }` 点新歌，或 `{ "job_id": "..." }` 复用已准备歌曲 |
| `POST /v1/room/commands` | `play`、`pause`、`seek`、`next`、`select`、`remove`、`reorder`、`mix`、`ready`、`retry` |
| `GET /v1/room/events` | 推送 `event: room`；空闲时只发连接保活注释 |
| `GET /v1/storage` | 缓存使用量、上限、受保护与可回收容量、准备空间是否足够 |
| `POST /v1/storage` | `{ "limit_gib": 10 }` 修改并持久化上限，返回新的房间快照 |

`pause`、`seek`、`next`、`ready` 必须带 `entry_id`，防止旧播放器事件影响新歌曲。`reorder` 传入完整的 `order` ID 数组，可带 `revision` 拒绝过期操作；当前歌曲保持首位。`mix` 接受 `instrumental`（纯伴奏）、`karaoke`（伴奏＋伴唱）、`original`（三轨一起播放）。

`levels` 独立调整三个声部，支持部分更新，例如 `{ "action": "levels", "levels": { "backing_vocals": 0.35, "lead_vocals": 0.2 } }`。键为 `instrumental`、`backing_vocals`、`lead_vocals`，值为 0–1 的有限数。房间快照中的 `levels` 包含全部三个值，手动设置不对应预设时 `mix` 为 `custom`。切歌、暂停、定位不改变音量；选择 `mix` 预设则重置三个值。`storage` 随房间 SSE 快照推送，前端不轮询。

`pitch` 实时设置三轨共同的升降调，例如 `{ "action": "pitch", "pitch_semitones": -2 }`。接受 -6 至 +6 的整数半音数，0 为原调；快照中的 `pitch_semitones` 持久保存，遥控页同步。此命令不改变播放位置、播放速度、音量、预处理结果或缓存音轨。

预处理队列按歌单中尚未开始的歌曲重新排序，不中断当前处理。播放抵达未准备好的歌曲时等待；媒体加载完成后由播放页发送 `ready`，从设定位置起播，不把预处理／加载耗时算入歌曲时间。移除点歌项解除缓存保护；已提交的预处理仍会完成，成品可按 LRU 回收。

浏览器用同一个 Web Audio 时钟播放三条 MP3 试听音轨，MV 静音跟随。普通页面负责播放；`/?remote=1` 只点歌和控制，不发声。完整说明见 [APP.md](APP.md)。

## 提交

```sh
curl -sS http://127.0.0.1:8746/v1/jobs \
  -H 'Content-Type: application/json' \
  -d '{"url":"https://www.youtube.com/watch?v=n5sjpCQhTDk"}'
```

也支持 `youtu.be`、YouTube shorts/live 的录播、Bilibili BV/av 地址、`?p=2` 等分 P，以及 `b23.tv` 短链。一个任务只处理一个视频；直播流、播放列表和超过默认 15 分钟的视频会拒绝处理。

立即返回 HTTP 202：

```json
{
  "id": "任务 UUID",
  "status": "queued",
  "active_stages": [],
  "status_url": "http://127.0.0.1:8746/v1/jobs/任务UUID",
  "cache_hit": false
}
```

### 可选参数

| 字段 | 默认 | 用途 |
|---|---|---|
| `mv_audio` | `instrumental_backing` | 成片使用背景音乐＋伴唱；设为 `instrumental` 只播放背景音乐 |
| `lyrics` | 自动 | 字符串数组或按换行分句的文本；应与该视频版本实际演唱一致 |
| `language` | 自动 | `zh` / `ja` / `en`，也接受 `Chinese` / `Japanese` / `English` |
| `title`、`artist` | 自动 | 覆盖片头歌曲和原唱信息 |
| `force` | `false` | 强制创建新任务，不复用同链接、参数、流水线版本的任务 |

```json
{
  "url": "https://www.bilibili.com/video/BV1bK411W797?p=1",
  "mv_audio": "instrumental",
  "language": "ja"
}
```

## 查询和下载

```sh
curl -sS http://127.0.0.1:8746/v1/jobs/任务UUID
```

状态依次为 `queued`、`running`、`completed`；失败为 `failed`，通过 `error.code` 与 `error.message` 获取原因。`active_stages` 会显示当前正在下载、分离、整理歌词、对齐还是渲染。同一首歌的独立步骤可并行，因此可能同时显示两个阶段。

`stage_progress` 按阶段提供实测计数，例如 `{"separate_music":{"completed":12,"total":40,"unit":"chunks","detail":null}}`。下载按视频／音频分别计字节，分离计已处理音频块，对齐计已处理段落／行，渲染计已送入编码器的帧。`total` 缺失时不计算百分比；加载模型、在线歌词请求、封装与校验仅提供阶段或 `detail`。这些是当前步骤的工作量，不能相加当作整首歌的时间百分比。只有成品通过校验后才会进入 `completed`。

房间快照也包含每首运行任务的 `stage_progress`，通过 SSE 更新。进度来自子进程的结构化标准输出；计数更新通常限频为每 0.5 秒一次，阶段切换立即推送。前端不轮询，不根据经过的时间伪造进度。

曲库从入队开始就包含歌曲，状态可以是 `queued`、`running`、`completed`、`failed` 或 `evicted`；静态目录中尚未提交的歌曲为 `unprepared`。同一视频只展示最近一次任务。`job_id` 也可指向尚未完成的任务，再次点播时复用该任务，不重复下载或处理；只有 `status == "completed"` 才表示已准备好。房间 SSE 快照的 `library` 与 `GET /v1/library` 使用同一份数据，元数据、语言分类和进度一并推送。

语言优先采用用户指定值，再使用已生成结果、当前音频语言信息或已知曲目目录。处理时先读取所选音频轨的语言标签，ASR 返回后立即更新，歌词整理后再次确认。未知语言暂时只列在「全部」，不会按标题文字或字幕语言猜测演唱语言。

取得下载元数据后，`song_info` 立即提供 `title`、`artist`、`duration_seconds`，并通过房间 SSE 更新歌单，不必等整首准备完成。优先使用视频的音乐标签，否则先显示视频标题；不把上传者当成原唱。歌词整理完成后再更新歌曲信息，用户填写的歌名／歌手始终优先。`input` 保持原请求，`result` 仍只在成品完成后提供。

被 LRU 回收的任务为 `evicted`，原文件接口返回 HTTP 410。重新提交同链接或用原 `job_id` 点歌会重新准备。缓存不足导致准备中断时错误码为 `cache_full`；不会删除待唱歌曲来掩盖空间不足。默认上限、统计范围和运行中写入限制见 [缓存说明](APP.md#本地缓存上限)。

完成后的 `result.artifacts` 每项包含 `url`、`bytes`、`sha256`：

| 名称 | 文件 |
|---|---|
| `mv` | 已烧录字幕的完整 MP4；混音含义见 `result.mv_audio` |
| `instrumental` | 背景音乐 WAV |
| `backing_vocals` | 伴唱 WAV |
| `lead_vocals` | 主唱 WAV |
| `instrumental_preview` | 背景音乐 MP3 |
| `backing_vocals_preview` | 伴唱 MP3 |
| `lead_vocals_preview` | 主唱 MP3 |
| `subtitles` | 包含逐字／词时间的 JSON |
| `poster` | 可选的本地 JPEG 封面：优先视频缩略图，获取失败时截取 MV 画面 |

文件接口支持 HTTP Range，可直接交给播放器或下载。处理完成前访问文件返回 409；未发布的内部文件无法通过文件接口读取。

```python
import time
import httpx

base = 'http://127.0.0.1:8746'
with httpx.Client(timeout=30) as client:
    response = client.post(base + '/v1/jobs', json={'url': 'https://www.youtube.com/watch?v=n5sjpCQhTDk'})
    response.raise_for_status()
    job = response.json()
    while job['status'] in ('queued', 'running'):
        time.sleep(2)
        response = client.get(job['status_url'])
        response.raise_for_status()
        job = response.json()
    if job['status'] != 'completed':
        raise RuntimeError(job['error'])
    for name, artifact in job['result']['artifacts'].items():
        print(name, artifact['url'])
```

`GET /v1/jobs?limit=20` 列出近期任务；`GET /health` 返回队列和正在处理的任务。

## 自动歌词与质量边界

已核对的本地歌词只在视频身份和时长匹配时复用。新视频从分离人声生成 ASR，再由文本模型整理歌词和歌曲信息。自动整理结果标为 `asr_llm_draft` / `needs_review`；它不是出版歌词的正确性保证。文本和这个版本的演唱匹配不足、粗时间轴冲突或字体缺字时，任务明确失败，可提供修正后的 `lyrics` 再提交。

逐句音素细化有时会让后一句提前越过前一句。接口会检查全局起点顺序，对冲突行回退到原有粗对齐，并通过 `quality.line_order_fallback_lines` 标记；不会手工平移某个字来凑顺序。粗对齐本身有冲突时不生成伪造的时间轴。

中文和日文沿用歌声音素与三窗口筛选；英文沿用歌声音素起点和 Whisper 近邻融合。所有语言都使用当前句＋下一句、琥珀色点亮、硬切、片头信息和四秒开唱提示。视频中的旧字幕不参与歌词识别或对齐。

母版是未缩放的 float64 WAV，按 `(主唱 + 伴唱) + 背景音乐` 合并验证原始 PCM。MP3 和 MV 使用同一个防削波增益。编码试听允许一个 MP3 帧以内的末尾静音填充；缺失音频或新增非静音样本仍失败。精确还原不等于分离纯度，也不证明亮字准确率；结果中这两项没有伪造分数。

## 配置

| 环境变量 | 说明 |
|---|---|
| `OPENAI_API_KEY` | ASR 和自动歌词整理需要 |
| `KARAOKE_TEXT_MODEL` | 默认 `gpt-5-mini`；仅负责元数据／歌词文本整理 |
| `KARAOKE_DATA_DIR` | 默认 `karaoke/service-data`，保存 SQLite 任务状态和每个任务的文件 |
| `KARAOKE_CACHE_MAX_GIB` | 初次创建缓存设置时的容量上限，默认 `10` GiB；之后以页面保存的值为准 |
| `KARAOKE_PORT` | 默认 `8746` |
| `KARAOKE_HOST` | 默认 `127.0.0.1`；需要局域网设备点歌时可显式设为 `0.0.0.0` |
| `KARAOKE_CORS_ORIGINS` | 允许调用接口的前端来源，逗号分隔；默认允许本地 8745 预览页 |
| `KARAOKE_DEVICE` | 默认 `auto`：CUDA → MPS → CPU；也可显式指定 |
| `KARAOKE_SEPARATOR_PYTHON` | 分离环境的 Python；本机默认复用已有 `audio/.venv-separator-local` |
| `KARAOKE_MAX_DURATION_SECONDS` | 默认 `900` |
| `KARAOKE_COOKIES_FILE` | 可选的 yt-dlp cookies 文件，用于需要登录的视频；不会自动读取浏览器 cookies |
| `KARAOKE_FONT_FILE` | 可选的完整字体路径；Linux 可使用覆盖对应语言的 Noto Serif CJK 字体 |
| `KARAOKE_DISABLE_LYRIC_CATALOG` | 设为 `1` 时强制走自动歌词，用于测试新歌流程 |

单进程、单 GPU 队列运行；不要给同一数据目录启动多个 Uvicorn worker。独立的分离环境与主环境都需安装对应硬件的 PyTorch。CUDA 入口已接入；本次实际端到端运行平台是 Mac / MPS，未在 RTX 4090 上测时。

服务重启后恢复排队任务；被中断的运行任务标为失败，保留日志和中间文件，不假装完成。失败任务可以重新提交；已完成且文件完整的相同请求直接命中缓存。

## 本次端到端验证（Mac / MPS）

| 输入 | 视频时长 | 处理总时长，不含排队 | 歌词来源 |
|---|---:|---:|---|
| YouTube《美丽的神话》 | 4 分 50 秒 | 6 分 04 秒 | 同版本已核对歌词 |
| Bilibili《Staple Stable》片段，含角色对话 | 1 分 30 秒 | 2 分 04 秒 | 音频识别＋自动整理，待复核 |

两次均通过 HTTP 提交真实 URL，下载、处理、渲染完整跑通；八个输出文件均通过完整下载哈希和首尾 Range 校验。三轨重组均与解码后的原始 PCM 逐样本一致。当时 68 项行为测试通过；这不是歌词听辨或切字准确率的测量。历史验证记录与媒体保留在原开发工作区，未随源码上传。
