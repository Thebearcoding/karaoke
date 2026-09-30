# 卡拉 OK 算法实验

> 这是按迭代过程积累的研究记录，包含早期状态与本机实验链接。当前应用和安装步骤以[仓库首页](../README.md)、[安装说明](../SETUP.md)及 [APP.md](APP.md) 为准。本文引用的音视频、数据集、试听页和实验结果不随源码上传；后文的历史状态不代表当前功能。

**网页点歌台「余音」：[使用说明](APP.md)。** 启动 `python -m karaoke.api` 后访问 `/`，支持链接点歌、后台准备、歌单排序、自动下一首、三种人声模式及遥控点歌。页面通过 SSE 接收状态。

**本地 URL 接口：[API 用法](API.md)。** 提交 YouTube / Bilibili 视频链接，后台生成带字幕 MV、背景音乐、伴唱、主唱，并提供进度查询、结果下载与缓存。默认成片使用背景音乐＋伴唱。

最新完整复跑：[《美丽的神话》三轨与当前字幕效果](experiments/shen-hua-v001/README.md) · [试听页](experiments/shen-hua-v001/index.html)。重新运行现有分离和音频对齐，提供伴奏＋伴唱、只放背景音乐、原唱核对三种同画面版本。

最新候选：[《Perfect》起音对齐迭代](experiments/perfect-v004/README.md)。保持当前句＋下一句、硬切和原音轨，更新逐词点亮时间。[六首人工标注回归结果](experiments/word-onset-v001/README.md)：时间戳 ±50 ms 准确率 28.3% → 54.6%，仍未达标且有个别退步。原双行展示版保留在 [v003](experiments/perfect-v003/index.html)。

首次新歌测试：[Ed Sheeran《Perfect》原唱／伴奏试听](experiments/perfect-v001/index.html) · [完整结果](experiments/perfect-v001/README.md)。当时沿用既有模型和英文词级对齐，没有手调时间戳；首次运行暴露 MP3 尾端填充校验问题，已保留失败记录并单独适配试听导出。当时尚无英文音素细化，两处合并点亮保留在该历史版本中。

最新局部修正：[《突然好想你》原唱对照](design/subtitles-v005/index.html) · [完整 MV 候选](experiments/karaoke-final-v006/README.md) · [重复歌词辅助对齐说明](experiments/word-timing-v002/README.md)。

分离阶段已保存为 [separation-f-v1 checkpoint](checkpoints/2026-09-18-separation-f-v1/README.md)，包含经过哈希核对的代码、四首双轨、原始音源和 F 模型权重。后续字幕工作在此基础上继续。

**当前研究基线是 F：Kimberley Jensen 的 Mel-Band RoFormer。** 用户已授权预训练 NN 效果对照，并在 Lemon 片段上反馈“F很牛，比E好，E也很好”。六段真实参考上的两项伴奏重建门槛全部通过；这不等于完整音质与产品验收通过，最终产品是否采用 NN 尚未决定。

最新结果见 [第三轮报告](experiments/iteration-003.md)，论文与公开权重见 [研究笔记](research/2026-09-18/README.md)，门槛见 [ACCEPTANCE.md](ACCEPTANCE.md)。F 处理完整 274.27 秒 Lemon 用时 95.49 秒（本机 M4 Max，纯分离计时）。

此前的 [第一轮](experiments/iteration-001.md) 与 [第二轮](experiments/iteration-002.md) 非 NN 分离候选均未通过全部质量门槛；这些旧实验的歌词预对齐也未达标。

**最新字幕：琥珀色跟唱提示已按歌声发音细化。** [四首新版完整 MV](experiments/karaoke-final-v005/README.md)使用背景音乐＋伴唱，另有[新旧时间轴对比](design/subtitles-v004/index.html)。保留较厚宋体、透明底与硬切换句；由原来的整词组点亮改为按音素定位到字或读音组合，异常处保守回退。方法和限制见[逐字对齐记录](experiments/word-timing-v001/README.md)，尚待实际跟唱验收。

[上一版完整 MV](experiments/karaoke-final-v002/README.md)包含原来的四首硬切字幕版本，以及《美丽的神话》保留孙楠、尝试去掉韩红的同曲模板试验。后者的合唱效果仍需试听确认，方法和诊断见 [duet-v001](experiments/duet-v001/README.md)。

最新的 [男女合唱分离对照](experiments/duet-chorus-v002/README.md)已分别导出《美丽的神话》男声、女声和合唱段试听，并保留模板版作同位置比较。

主唱／伴唱三轨任务的[候选方案](experiments/three-stems-v001/RESEARCH.md)和[评测口径](docs/lead-backing-metrics.md)单独记录。没有真实主唱／伴唱参考轨时，bitwise 还原、音量比例和歌词识别率都不能替代分离纯度评测。

**四首完整三轨候选：[主唱、伴唱、背景音乐与保留伴唱的 MV](experiments/three-stems-v001/README.md)。** 使用已有 F 人声加 Anvuew Karaoke 模型，保留原背景音乐母版；另有 Frazer/Becruily、直接处理原曲及波形平均的同段对照。四首新增分离各约 83–112 秒，按指定浮点求和顺序均通过原始 PCM 的 bitwise 还原。尚待主唱／伴唱纯度试听验收。

## 当前内容

- `samples/`：Lemon、晴天、群青、突然好想你现场版的固定原始输入。
- `dsp.py`：无神经网络的空间滤波、重复背景估计、低秩分解、谐波跟踪和立体声回归候选。
- `align.py`：根据音节长度与声学活动寻找句子边界的基线。
- `phonetic_align.py`：规则发音合成模板与单调序列匹配。
- `gaussian_align.py`：只在校准歌曲上拟合的非神经网络高斯发音模板。
- `benchmark_*.py`：各轮冻结的实验流程；测试参考时间与分轨不进入产品推理。
- `experiments/listening-v002/`：四首的完整原曲/候选音频，以及短对比试听。
- `experiments/listening-v003/`：原 A 与更强候选 D 的同位置对比，以及各首完整 D；这些仍是未通过验收的诊断候选。
- `experiments/neural-v001/`：E/F/G 三种预训练权重与 A/C/D 的同输入对照、原始浮点声部、时延、用户反馈和验证记录。
- `experiments/listening-v004/`：四首完整 F 伴奏、Lemon 人声持续片段的原曲/C/F/G 对比，以及另外三首的原曲/C/F 对比。
- `experiments/environment.json`、`source-manifest.json`：运行环境与代码校验信息。
- `data/*-manifest.json`：下载的参照数据 SHA-256。

这些是算法实验模块；尚未接入搜索界面、Codex 备歌服务或完整播放器。

**当前字幕输入约定：假设 MV 没有歌词，也没有可用的字幕时间轴。** 用经过核对的纯文本歌词和分离干声，在预处理阶段预测逐句时间；视频只负责播放背景。画面 OCR 和其他版本的字幕时间不进入这条流程。用户已允许语音模型，音频对齐的质量仍待迭代。

**原四首的硬切字幕成片：[完整卡拉 OK MV](experiments/karaoke-final-v002/README.md)。** 使用 F 伴奏、用户选定的 A · 余韵字幕，以及由干声和纯文本歌词计算的逐句时间轴，共 216 句。ASR 先定位段落，Whisper large-v3-turbo + stable-ts 做强制对齐；繁简规范化仅用于匹配，显示保留原歌词。时间轴、显示提前量、成片和验证记录分别保存。这一版已完成全片解码和前中后抽查，尚未声称通过人工标注的时间误差门槛。

[四首第一版完整预览](experiments/subtitle-v001/README.md)保留为历史混合基线：它曾结合画面文字、现有字幕和 ASR，因此不能证明当前无字幕输入约定已经通过。Lemon 的纯音频候选亦尚未完成逐句时间戳验收。

字幕视觉方案见 [设计说明](docs/design/subtitles.md) 和 [最新动效预览](design/subtitles-v004/index.html)。第一版用户选定 A · 余韵（`cinema`），第二版默认 `cinema_focus` 在此基础上改善字号、字重和对比度。`align_lyric_phonemes.py` 细化句内点亮时间，`render_lyric_progress.py` 支持 `--mode word|underline|static`；当前字幕按用户选择直接切换，不渐入渐出。默认样式、字体、大小、字距与动效时长集中保存在 `subtitle-styles.json`。[第一版样式预览](design/subtitles-v001/index.html)保留用于比较。

## 试听

### 最新：四首主唱、伴唱和背景音乐

[三轨文件、MV 与对照](experiments/three-stems-v001/README.md)包含完整 MP3、未缩放 WAV 母版及验证记录；新 MV 使用“背景音乐＋伴唱”，沿用 A 样式、音频对齐和硬切字幕。

### 四首完整人声与互补伴奏

[双轨文件与验证结果](experiments/stems-v001/README.md)：四首均已导出完整人声、伴奏，以及方便试听的 MP3。母版采用原始 48 kHz 的 64 位浮点 WAV；实际文件以 64 位相加，与原录音解码后的 PCM 逐比特一致。少量串音按用户最新反馈可接受，听感纯度与数值还原分别判断。

### 最新：预训练模型效果对照

- [Lemon 完整 F 伴奏](experiments/listening-v004/lemon-F-full.mp3) / [无损 FLAC](experiments/listening-v004/lemon-F-full.flac)。
- [晴天完整 F 伴奏](experiments/listening-v004/qing-tian/F-full.mp3) / [无损 FLAC](experiments/listening-v004/qing-tian/F-full.flac)。
- [群青完整 F 伴奏](experiments/listening-v004/gunjou/F-full.mp3) / [无损 FLAC](experiments/listening-v004/gunjou/F-full.flac)。
- [突然好想你现场版完整 F 伴奏](experiments/listening-v004/suddenly-missing-you-live/F-full.mp3) / [无损 FLAC](experiments/listening-v004/suddenly-missing-you-live/F-full.flac)。
- [Lemon 原曲 → C → F → G](experiments/listening-v004/comparison.mp3)：原曲 104.28 秒处，各 12 秒，间隔 0.5 秒。
- [晴天](experiments/listening-v004/qing-tian/comparison-RCF.mp3)、[群青](experiments/listening-v004/gunjou/comparison-RCF.mp3)、[突然好想你现场版](experiments/listening-v004/suddenly-missing-you-live/comparison-RCF.mp3)：原曲 → C → F，各 12 秒。用户给出正面初评并要求完整歌曲，反馈保存在 [feedback.json](experiments/listening-v004/feedback.json)。

### 之前：非 NN 候选

每个 `comparison.mp3` 依次为 **原曲 R → A → B → C**，每段 12 秒，中间停 0.5 秒。四段使用同一原曲位置、同一导出增益。完整音频分别为 `R.flac`、`A.flac`、`B.flac`、`C.flac`。

- [Lemon](experiments/listening-v002/lemon/comparison.mp3)
- [晴天](experiments/listening-v002/qing-tian/comparison.mp3)
- [群青](experiments/listening-v002/gunjou/comparison.mp3)
- [突然好想你现场版](experiments/listening-v002/suddenly-missing-you-live/comparison.mp3)

算法对应表位于 `experiments/listening-v002/answer-key.json`。用户对 Lemon 的初评为 A > C > B，但进一步指出 A、C 均未去掉原唱，B 又削弱了伴奏。A 仅作为低损伤对照，不能视为分离成功；原始反馈保存在 `experiments/listening-v002/feedback.json`。

## 在当前工作目录运行

```sh
.venv/bin/python -m unittest discover -s karaoke -p 'test_algorithms.py' -v

OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 \
  .venv/bin/python karaoke/benchmark_separation.py \
  --output karaoke/experiments/separation-recheck --count 12

OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 \
  .venv/bin/python karaoke/benchmark_gaussian.py \
  --output karaoke/experiments/alignment-recheck
```

其他 benchmark 脚本同样支持 `--output`，默认拒绝覆盖已有实验结果。后续再看过的留出歌曲应视为开发材料，不能在其上继续调参后再称作盲测。

第二轮的脚本为 `benchmark_spatial_search.py`、`benchmark_spatial_coupled.py` 和 `render_spatial_followup.py`。各实验目录保存对应运行时的代码快照；根目录代码继续演进时，以快照解释历史数据。

非 NN 实验依赖列在 `requirements.txt`，字幕依赖列在 `requirements-subtitles.txt`。eSpeak NG 是规则/共振峰合成器；高斯模板是经典统计模型。第三轮预训练模型另用现有的独立音频环境，复现细节见研究笔记。早期非 NN 发音与对齐实验只验证英文；当前中日文成片采用独立的语音模型路线，不代表旧算法通过了中日文验收。
