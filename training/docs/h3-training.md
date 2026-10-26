# H3 dense 音视频训练

当前提供 MiniMax H3 的音视频联合 **DMD 与分段端点 TDM**，采用全注意力、FSDP2、student Attention＋FFN LoRA、全参数 fake，以及离线 prompt 条件。此入口没有稀疏注意力；`flash_4` 是 dense attention 后端，不是 ASA／VSA。

代码使用 Diffusers 原生 H3 模型、布局和 VAE，并封装训练与音视频采样循环；未直接执行 MiniMax 或 LightX2V 的完整训练器。当前示例为文本到音视频，不承诺支持官方所有多模态任务。

## 环境与模型

使用独立环境，建议 Python 3.12 与 PyTorch 2.9.1。FSDP2 需要多卡和充足显存，参考启动器默认 8 个进程，可用 `NPROC_PER_NODE` 调整。显存取决于 world size、分辨率、帧数和重计算策略，先执行两步预检。

```bash
python -m venv .venv-h3
source .venv-h3/bin/activate
# 先按 PyTorch 官方说明安装匹配硬件的 CUDA 构建。
python -m pip install -r requirements/dense.txt
python -m pip install --no-deps -e .
# 默认参考配置使用 Blackwell FA4／CUDA 13 的可选后端。
python -m pip install -r requirements/h3.txt

hf download MiniMaxAI/MiniMax-H3 \
  --revision 42ed227ee7df40d41602854ae760620d6eb651fe \
  --local-dir models/MiniMax-H3
```

模型来自 [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)，遵循其 Community License；须自行满足上游下载和使用条件。源码依赖和模型权重的许可分别适用，本仓库不包含权重、缓存或 token。

参考配置 `attention_backend=flash_4` 需要匹配的硬件和构建。也支持 `_native_cudnn` 与 `native`；更换后端应单独进行显存和数值预检。CUDA 13 的 FA4 可选依赖并非所有 CUDA GPU 的通用安装方案。

## 离线文本条件

原生 H3 条件来自自带 tokenizer 和 Qwen3-VL 第 50 层，不可使用 Wan 的 UMT5 特征。编码时不加 chat template 或 special tokens，条件为变长序列，不做 padding；训练和固定验证不驻留 text encoder。

```bash
python -m video_blade prepare-h3 \
  --model models/MiniMax-H3 \
  --prompts data/prompts/example_train.txt \
  --output outputs/prompts-h3-train

python -m video_blade prepare-h3 \
  --model models/MiniMax-H3 \
  --prompts data/prompts/example_validation.txt \
  --output outputs/prompts-h3-validation
```

离线编码本身需要能容纳文本编码器的 GPU；也可通过 `torchrun -m video_blade.h3.prepare` 在多卡上分配不同 prompts。示例文本用于接口检查，正式实验需使用来源明确、规模足够的数据和独立验证集。

可选 `python -m video_blade import-h3-cache` 支持固定版本的 [公开 VidProM H3 条件](https://huggingface.co/datasets/pdmd2026/rcm-vidprom-h3-qwenvl-cache)。导入会下载并校验完整缓存，体积较大；使用前查看发布方说明并确认存储空间。若采用此数据，将配置的 `embeddings` 改为导入目录，验证条件仍需独立准备。

## 预检与正式训练

```bash
# 两次 student、十次 fake 更新，只导出 student adapter。
bash scripts/train_h3.sh --config configs/h3_tdm_preflight.json

# 正式 TDM，默认最多 300 次 student 更新。
bash scripts/train_h3.sh --config configs/h3_tdm.json

# 独立 DMD 基线；参考配置最多 1000 次 student 更新。
bash scripts/train_h3.sh --config configs/h3_dmd.json
```

配置采用 1344×768、124 帧、24 fps，student 为 4 次前向、video/audio shift 6／3，teacher 为 49 次前向、shift 12／3，CFG 均为 1。H3 速度方向与 Wan 不同，适配层已转换；不得将 Wan 的 1000 倍时间或单时间表套入 H3。

Student LoRA rank 128／alpha 8；fake 全参数。AdamW 学习率固定为 `5e-5/4e-7`，每轮先 1 次 student、后 5 次 fake，各次取新样本。TDM 使用分段端点的 normalized L2，视频与音频分别归一化后求和。DMD 和 TDM 属于不同实验目标，恢复检查会拒绝跨算法状态；本实现的 TDM 时间权重属于 H3 适配配方，不能称为上游 demo 的逐项复现。

Teacher／student／fake 使用 FSDP2；分段退出步在 rank 间同步，以匹配 collective 次数。训练中的 dtype、RoPE 重建及 activation checkpointing 由适配层管理；部署时应预检，不能将几步更新成功当作质量或长期稳定性结论。

## 保存、恢复与独立采样

每 50 步导出 LoRA 和验证音视频，每 100 步保存完整 DCP 状态。完整状态包括全参数 fake 和两份优化器，可能占用数百 GB；规划存储容量，不将其提交到 Git。

```bash
bash scripts/train_h3.sh --config configs/h3_tdm.json \
  --resume outputs/h3-tdm/checkpoint-000100

# 采样同样使用 torchrun；只生成指定 checkpoint 的样本。
torchrun --standalone --nproc_per_node=8 -m video_blade sample-h3 \
  --config configs/h3_tdm.json \
  --adapter outputs/h3-tdm/adapters/step-000200 \
  --step 200 --output outputs/h3-sample
```

完整 `checkpoint-*` 可恢复训练；`adapters/step-*` 标记 `resumable=false`，只适合独立采样。独立采样从 `--config` 读取模型路径、rank／alpha、时间表及 `validation_embeddings`；这些字段须匹配所选 adapter，实际保存步数和采样参数以配置及输出 metadata 为准。已有输出不会被隐式覆盖，训练进度查看 `metrics.jsonl` 和 `stages.jsonl`。

## 方法与上游组件

- [MiniMax H3](https://github.com/MiniMax-AI/MiniMax-H3)：模型、原生组件与模型条款。
- [Diffusers](https://github.com/huggingface/diffusers)：Transformer、scheduler、VAE 和 packed layout。
- [LightX2V H3 配方](https://github.com/ModelTC/LightX2V/blob/540102c31eae93ed4ada01647144de9d9045fde3/lightx2v_train/configs/train/dmd/minimax_h3_t2av_dmd_lora.yaml)：DMD 优化器、LoRA 与采样设置参考。
- [TDM](https://arxiv.org/abs/2503.06674)：分段轨迹分布匹配思想参考；音视频双时间表、归一化与显式权重是本实现的适配。
