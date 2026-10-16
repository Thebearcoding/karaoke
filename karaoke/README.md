# 源码导航

使用项目先看根目录的 [README](../README.md)、[安装说明](../SETUP.md) 和 [网页指南](APP.md)。
服务入口为 `python -m karaoke.api`；前端不需要构建。歌曲、歌词、模型和运行数据不随源码提供。

## 正式处理路径

| 模块 | 职责 |
| --- | --- |
| `api.py`、`room.py`、`job_schema.py` | HTTP / SSE、单房间歌单、任务状态与请求校验 |
| `cache.py`、`stage_cache.py`、`stage_progress.py` | 容量回收、检查点与阶段进度 |
| `pipeline.py` | 下载、分离、歌词、对齐、封装和校验的调度 |
| `song_sources.py`、`lyric_*.py`、`local_asr.py` | 来源身份、歌词检索、文本证据与识别 |
| `benchmark_neural.py` | 正式流程使用的分离子进程入口；名字沿用早期实验 |
| `experiments/cjk-onsets-v003/run.py` | 正式流程使用的中日文多窗口对齐，目录名沿用早期实验 |
| `align_*.py`、`phoneme_progress.py`、`timing_validation.py` | 声学对齐、字词位置和时间检查 |
| `exact_stems.py`、`three_stems.py` | 音轨导出、共同增益与完整性检查 |
| `subtitle*.py`、`render_*.py` | 字幕布局、高亮和视频合成 |
| `web/` | 原生 HTML / CSS / JavaScript 播放器 |

[处理与存储流程](docs/processing.md)说明阶段依赖；[API](API.md)说明请求、产物和运行配置。
普通播放保存底轨（背景音乐＋伴唱）与主唱两条 FLAC；严格分轨验收使用独立的三轨产物。

## 评测与维护

- `tools/quality.py`：读取独立参考，评估起音和分轨质量；见[质量检查](docs/automatic-quality.md)。
- `tools/lyric_audit.py`：歌词来源与正文诊断；见[歌词检索审计](docs/lyric-retrieval-audit.md)。
- `tools/soak.py`、`tools/playback_sweep.py`：隔离数据目录的准备与播放回归；见[回归流程](docs/soak-testing.md)。
- 其他 `benchmark_*.py`、`dsp.py`、`gaussian_align.py`、`duet_templates.py` 保留为算法研究工具，需要自行准备对应数据和可选依赖，不属于默认服务启动路径。
- `dependencies/` 保存第三方来源与哈希，`models/pretrained/` 保存模型配置。许可范围见[第三方说明](../THIRD_PARTY.md)。

[贡献指南](../CONTRIBUTING.md)列出测试与提交要求；音质和起音准确率的测量方法见[验收标准](ACCEPTANCE.md)。
