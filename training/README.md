# 当前训练实现

此目录是当前维护的训练代码，所有命令以 `training/` 为工作目录，模型默认位于 `models/`，训练输出位于 `outputs/`。

- `video_blade/`：公共配置、数据、flow、模型、DDP Wan2.1 dense 训练与采样。
- `video_blade/h3/`：H3 音视频联合 dense DMD／TDM、FSDP2、checkpoint 与采样。
- `configs/`、`scripts/`、`requirements/`：示例配方、通用启动器和独立依赖。
- `data/prompts/`、`tests/`、`docs/`：示例文本、回归测试和公开使用说明。

安装后使用 `python -m video_blade --help` 查看入口。完整步骤见 [dense Wan2.1](docs/dense-training.md) 和 [H3 训练](docs/h3-training.md)。原 BLADE 稀疏实现位于同级 [legacy](../legacy/README_zh.md)，不要混用两套依赖环境。

## 稀疏注意力计划

当前训练使用 dense attention，包括 `flash_4` 后端。后续计划更新稀疏注意力方式，并逐步接入当前训练框架和 H3；具体配方与质量／效率结果将在验证完成后发布。当前没有启用 H3 稀疏训练。

仓库整体结构、原论文和引用信息见[主 README](../README_zh.md)。
