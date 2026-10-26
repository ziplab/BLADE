# BLADE: Block-Sparse Attention Meets Step Distillation for Efficient Video Generation

[论文](https://arxiv.org/abs/2508.10774) · [项目主页](http://ziplab.co/BLADE-Homepage/) · [模型](https://huggingface.co/GYP666/BLADE) · [English](README.md)

仓库按当前实现和历史实现分为两个目录：`training/` 放当前训练代码，`legacy/` 保留原 BLADE 实现与论文复现入口。

## 目录结构

```text
BLADE/
├── training/                  # 当前训练实现
│   ├── video_blade/           # 共享代码、dense Wan2.1、H3 DMD/TDM
│   ├── configs/               # 可移植示例配方
│   ├── scripts/               # 分布式启动与源码检查
│   ├── requirements/          # 当前依赖
│   ├── data/prompts/          # 示例提示词
│   ├── tests/                 # 回归测试
│   ├── docs/                  # 训练与采样说明
│   └── pyproject.toml
├── legacy/                    # 原有实现
│   ├── cogvideox/             # CogVideoX 稀疏训练
│   ├── wanx/                  # Wan2.1 稀疏训练
│   ├── utils/                 # 原提示词预处理
│   ├── requirements.txt
│   └── README.md
└── README.md
```

| 目录 | 支持范围 | 注意力 |
| --- | --- | --- |
| `training/` | Wan2.1 dense 步数蒸馏；H3 音视频联合 DMD／TDM | Dense |
| `legacy/` | 原 CogVideoX、Wan2.1 BLADE 训练与推理入口 | 原自适应稀疏注意力 |

H3 当前为全注意力训练。`flash_4` 仅是 dense attention 后端，不代表已经支持稀疏注意力。当前 dense 配方与原论文稀疏实验使用各自入口。

## 当前训练

安装、准备条件、训练和采样都在 `training/` 下执行，使用与历史实现分开的环境。

```bash
cd training
# 先安装匹配硬件和 CUDA 的 PyTorch 2.9.1。
python -m pip install -r requirements/dense.txt
python -m pip install --no-deps -e .
# 参考 H3 配置使用 flash_4，需要其可选依赖。
python -m pip install -r requirements/h3.txt

python -m video_blade --help
bash scripts/train_h3.sh --config configs/h3_tdm.json
```

训练前先准备模型文件和 prompt embedding。完整流程见[当前训练概览](training/README.md)、[H3 训练](training/docs/h3-training.md)和 [dense Wan2.1 训练](training/docs/dense-training.md)。仓库只提供代码与示例配置，不包含模型权重或 prompt 缓存。

## 原有实现

复现原论文的 CogVideoX／Wan2.1 稀疏训练时，进入 `legacy/`，按[原英文说明](legacy/README.md)或[原中文说明](legacy/README_zh.md)操作。其中路径以 `legacy/` 为工作目录。原训练目标与稀疏适配模块保留；旧本机路径和 Notebook 自动备份已清理。

## 后续稀疏注意力计划

- [x] 将当前实现与原有实现分目录整理。
- [x] 提供 dense Wan2.1 和 H3 音视频训练。
- [ ] 更新适用于当前训练框架的稀疏注意力方式和计算内核。
- [ ] 接入并验证 H3 稀疏注意力训练。
- [ ] 发布经过验证的训练配方，以及同设置下 dense／sparse 的质量和效率对照。

后续会逐步把更新的稀疏注意力方式接入当前训练框架与 H3；上述未勾选内容为计划，不表示当前 `training/` 已经实现。

## 引用

```bibtex
@misc{gu2025bladeblocksparseattentionmeets,
  title={BLADE: Block-Sparse Attention Meets Step Distillation for Efficient Video Generation},
  author={Youping Gu and Xiaolong Li and Yuhao Hu and Minqi Chen and Bohan Zhuang},
  year={2025},
  eprint={2508.10774},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2508.10774}
}
```
