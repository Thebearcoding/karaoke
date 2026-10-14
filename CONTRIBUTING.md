# 参与开发

先按 [SETUP](SETUP.md) 准备环境。[源码导航](karaoke/README.md)区分服务入口、正式处理阶段和研究工具；项目约束见 [AGENTS.md](AGENTS.md)。

## 修改与验证

- 围绕具体问题提交小改动，说明触发条件、修改后的行为和验证结果。优先修复共同机制，避免歌曲特例。
- 扩展已有行为测试，覆盖实际故障和关键边界；静态文本检查不能替代行为或音质验证。
- 在独立、空的数据目录验证服务启动。不要让测试连接正在使用的房间、修改用户歌单或触发真实下载。
- UI 改动检查桌面、手机和全屏；音频／字幕改动遵循[验收标准](karaoke/ACCEPTANCE.md)。报告完整失败样本，不把模型置信度或时间单调当作起音准确率。

完整回归命令：

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=karaoke .venv/bin/python -m pytest -q karaoke/test_api.py karaoke/test_room.py karaoke/test_algorithms.py
node --test karaoke/test_player.mjs karaoke/test_pitch.mjs karaoke/test_search.mjs
```

测试需要主环境依赖和 SOFA / pyopenjtalk 发音字典，不需要歌曲库、模型权重或 API key。CI 运行同一组测试及空数据启动检查。

## 数据与凭据

歌曲、歌词库、封面、模型权重、生成文件、数据库、cookies、API key、个人部署信息和运行报告留在仓库外或 Git 已忽略的数据目录。测试优先使用合成音频与自写短文本；确需第三方素材时先核对分发授权。不要用 `git add -f` 绕过排除规则。

提交前检查 `git diff --cached --stat` 和 `git diff --cached`。发布前对整个 Git 历史做密钥扫描，例如安装 Gitleaks 后运行：

```sh
gitleaks git . --log-opts=--all --redact=100
```

扫描通过不代表没有个人信息。删除当前文件也不会删除它的 Git 历史；发现泄露时先撤销／轮换凭据，再与维护者安排历史处理。提交问题报告时去掉令牌、cookies、私人地址和完整歌词。

## 运行边界与许可

服务默认只监听本机，没有内置账号隔离；允许访问的用户可以操作共享歌单。远程访问应由部署者保护整个站点、API、媒体及 SSE，不能只保护首页。

项目自身许可证尚未选定；发布前需由维护者明确授权范围。新增依赖、复制的代码和模型配置须保留出处及许可，见 [THIRD_PARTY.md](THIRD_PARTY.md)。
