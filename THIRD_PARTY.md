# 第三方来源与许可

项目自身许可证尚未选定。以下是第三方组件的来源和已核对的许可，不能把它们统一解释为本项目的授权。模型、媒体和歌词各有独立的使用与分发条件。

## 随仓库分发的代码和配置

| 内容 | 来源与版本 | 许可与本地修改 |
| --- | --- | --- |
| `karaoke/web/vendor/signalsmith-stretch.mjs` | [Signalsmith Stretch](https://github.com/Signalsmith-Audio/signalsmith-stretch/tree/a670068d9aeb64913331d5cc29337b19a457a7df)，Geraint Luff / Signalsmith Audio Ltd. | [MIT 全文](karaoke/web/vendor/signalsmith-stretch.LICENSE.txt)。停用时跳过 WASM DSP，由原生延迟通道处理原调播放；修复声道数变化时的重新配置调用，并删除不可达的旧停用分支。原版和本地哈希见 [pitch-shift.json](karaoke/dependencies/pitch-shift.json)。 |
| `karaoke/benchmark_mimo.py` 的分块与交叉淡化实现 | [Sony MIMO 的 evaluate.py](https://github.com/SonyResearch/mimo-audio-separation/blob/9169bbbf9e6fdca5b22d5735b4244fbad7ab40b5/src/evaluate.py)，Yukara Ikemiya | 改编用于本项目的离线评测；保留 [MIT 全文](licenses/mimo-MIT.txt)。模型与上游源码不随仓库提供。 |
| `karaoke/models/pretrained/vocals_mel_band_roformer*.yaml` | [KimberleyJSN / Mel-Band RoFormer](https://huggingface.co/KimberleyJSN/melbandroformer) | 上游模型卡声明 MIT。保留上游配置及本项目的推理配置；权重哈希见 [separation.json](karaoke/dependencies/separation.json)。 |
| `karaoke/models/pretrained/karaoke_bs_roformer_anvuew.yaml` | [Anvuew 的固定版本](https://huggingface.co/anvuew/karaoke_bs_roformer/tree/0d4423d42e12cf2ba39ae09171028507b8a2a7be) | 该版本模型卡声明 `gpl-3.0`，保留 [GPL v3 全文](licenses/GPL-3.0.txt)。此声明不能被项目自身的许可证覆盖。 |

## 另行安装的源码与模型

- **audio-separator 0.44.5**：调用入口的[上游许可为 MIT](https://github.com/nomadkaraoke/python-audio-separator/blob/v0.44.5/LICENSE)；它使用的每个模型仍需单独核对。
- **SOFA**：固定源码版本的[许可为 MIT](https://github.com/qiuqiao/SOFA/blob/9549c6a86d16019c817eefe4bb8183405da524cc/LICENSE)。这不自动覆盖来自其他作者的日文／英文模型、字典或训练数据。
- **LyricsAlignment-Multilingual**：固定源码版本的[许可为 MIT](https://github.com/jhuang448/LyricsAlignment-Multilingual/blob/ca1a3923d6c8bf7d20eefa1080f3f104a6173d38/LICENSE)。来源与校验值见 [english-alignment.json](karaoke/dependencies/english-alignment.json)。
- **MMS_FA**：可选的声学复核模型；[PyTorch 官方文档](https://docs.pytorch.org/audio/stable/generated/torchaudio.pipelines.MMS_FA.html)明确声明 **CC-BY-NC 4.0（非商业）**。它不是通用的商业使用授权；如需商业部署，应另行取得适用授权或替换该模型。
- 其他 SOFA 权重、SenseVoice、Whisper、Qwen 等按 [SETUP](SETUP.md) 从原发布者取得；[依赖清单](karaoke/dependencies)记录版本和哈希。哈希只核对文件身份，不证明使用或再分发权限。
- 早期对照模型的来源可能没有完整许可声明，因此不随仓库提供其权重或配置，也不作为默认安装依赖。

Python 包由各 `requirements*.txt` 固定版本，Node.js、FFmpeg、LAME 和系统字体由部署者安装。这里没有对全部间接依赖或组合分发作兼容性结论；制作包含依赖／模型的容器、安装包或公开演示服务前，应核对所含组件的实际许可。

## 音乐和在线来源

本仓库不授予任何歌曲、MV、封面或歌词的版权许可。请使用有权处理和展示的素材，并遵守实际来源的访问条件。不要将私人 cookies、API key、下载媒体、歌词库或第三方研究 checkout 加入源码发布包。
