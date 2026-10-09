# 普通点歌：处理与存储

蓝色：成功后保留。橙色：处理中使用，成功后回收。箭头表示数据输入。

```mermaid
flowchart TD
  S["原始下载 source.mkv<br/>平台音频编码及原始采样率"]:::keep
  D["模型输入 source-44100.wav<br/>44.1 kHz · 双声道 · float32"]:::temp
  V["第一轮分离 vocals.wav<br/>完整人声 · 44.1 kHz · float32"]:::temp
  L["第二轮分离 vocals.wav<br/>主唱 · 44.1 kHz · float32"]:::temp
  A["对齐输入 vocals-16k.wav<br/>16 kHz · 单声道 · PCM16"]:::temp
  T["可信歌词 lyrics.json"]:::keep
  C["逐字时间轴 cues.json"]:::keep
  W["原采样率三声部计算<br/>临时 WAV · float64"]:::temp
  B["合并背景音乐＋伴唱<br/>共同增益 · accompaniment.flac<br/>原采样率和声道 · PCM24"]:::keep
  F["主唱 lead.flac<br/>相同共同增益<br/>原采样率和声道 · PCM24"]:::keep
  P["下载视频／封面画面"]:::temp
  M["字幕画面 mv.mp4<br/>仅视频，不含音轨"]:::keep
  S --> D --> V --> L
  V --> A --> C
  T --> C
  S --> W
  V --> W
  L --> W
  W --> B
  W --> F
  P --> M
  C --> M
  classDef keep fill:#deedfb,stroke:#2471b8,color:#123957
  classDef temp fill:#fff0da,stroke:#df8b29,color:#683e13
```

播放／预览：底轨＋主唱百分比 × 主唱轨，共用一个音频时钟，静音视频跟随。最终成品与歌词来源、必要诊断保留；两个无消费者的分离补轨不生成。失败待重试保留有用检查点，成功后不保留分离 WAV。再次点歌直接复用成品，只有明确重新准备才重跑处理。

独立严格音频验收仍保留三轨 float64 WAV、三轨 MP3 与字幕，不使用普通播放的两轨存储布局。
