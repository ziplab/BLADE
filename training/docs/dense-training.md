# Dense Wan2.1 训练

`video_blade` 为独立 dense 步数蒸馏实现，支持原生 PyTorch DDP、离线文本条件、LoRA、恢复和验证采样。它与 `wanx/` 中保留的原 BLADE 稀疏训练入口分别使用，不应将 dense 结果称为稀疏注意力复现。CogVideoX 仍使用原 `cogvideox/` 入口。

## 环境与模型

建议 Python 3.12，PyTorch `>=2.9.1,<2.10`，CUDA GPU；先安装与硬件匹配的 PyTorch，再安装其余依赖。避免将新旧依赖混入同一环境。

```bash
python -m venv .venv-dense
source .venv-dense/bin/activate
# 按 PyTorch 官方说明安装匹配 CUDA 的 torch 后继续。
python -m pip install -r requirements/dense.txt
python -m pip install --no-deps -e .
hf download Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --local-dir models/Wan2.1-T2V-1.3B-Diffusers
```

模型权重来自 [Wan 官方发布](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers)，使用和分发遵循其上游条款。本仓库不包含权重。

## 准备条件与训练

```bash
python -m video_blade prepare \
  --model models/Wan2.1-T2V-1.3B-Diffusers \
  --prompts data/prompts/example_train.txt \
  --output outputs/prompts-wan21-train --batch-size 8

python -m video_blade prepare \
  --model models/Wan2.1-T2V-1.3B-Diffusers \
  --prompts data/prompts/example_validation.txt \
  --output outputs/prompts-wan21-validation --batch-size 8

bash scripts/train_dense.sh --config configs/wan21_dense.json
```

示例 prompt 只用于跑通接口，应替换为适当规模、来源明确的训练文本，验证集需独立。条件缓存绑定模型和 Transformer 配置，不能仅修改 manifest 冒充另一模型的缓存。

参考配置使用 8 步、LoRA rank／alpha 64、teacher CFG 5、student 推理 CFG 1、500 次更新。每卡 batch 2、8 个进程时有效 batch 为 16；显存不足时可减小每卡 batch，按需增加累积并启用激活重计算，相关配置应重新预检。配置为示例，不保证适用于所有 GPU。

启动器默认使用当前环境的 `python`；可用 `VIDEO_BLADE_PYTHON` 指定解释器，`NPROC_PER_NODE` 指定可见 GPU 数。未调用个人环境脚本。模型、条件与输出路径均可在 JSON 中设置。

## 恢复与采样

```bash
bash scripts/train_dense.sh --config configs/wan21_dense.json \
  --resume outputs/wan21-dense/checkpoint-000100

python -m video_blade sample \
  --checkpoint outputs/wan21-dense/checkpoint-000500 \
  --embeddings outputs/prompts-wan21-validation \
  --sampler unipc --seed 8888 --output outputs/wan21-sample.mp4
```

实际 checkpoint 名称以输出为准。已有训练输出需要显式恢复或更换目录。恢复要求模型、数据 manifest、优化器相关配置一致；cosine 学习率恢复须保留原 `max_steps`。

独立采样默认 UniPC；训练配置的验证采样器会明确记录在视频旁的 JSON 中。随机 `trajectory` 与 UniPC 的轨迹不同，质量对照应固定 prompt、seed 和采样设置，不混用两者。
