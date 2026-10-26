# BLADE: Block-Sparse Attention Meets Step Distillation for Efficient Video Generation

[Paper](https://arxiv.org/abs/2508.10774) · [Project page](http://ziplab.co/BLADE-Homepage/) · [Models](https://huggingface.co/GYP666/BLADE) · [中文说明](README_zh.md)

The repository separates the current training implementation from the original BLADE implementation. Original paper experiments remain available in `legacy/`; new training support lives in `training/`.

## Repository structure

```text
BLADE/
├── training/                  # Current training implementation
│   ├── video_blade/           # Shared code, dense Wan2.1, and H3 DMD/TDM
│   ├── configs/               # Portable example recipes
│   ├── scripts/               # Launchers and source checks
│   ├── requirements/          # Current dependencies
│   ├── data/prompts/          # Example text prompts
│   ├── tests/                 # Regression tests
│   ├── docs/                  # Training and sampling guides
│   └── pyproject.toml
├── legacy/                    # Original implementation
│   ├── cogvideox/             # CogVideoX sparse-attention training
│   ├── wanx/                  # Wan2.1 sparse-attention training
│   ├── utils/                 # Original prompt preparation
│   ├── requirements.txt
│   └── README.md
└── README.md
```

| Directory | Supported path | Attention |
| --- | --- | --- |
| `training/` | Wan2.1 dense distillation; H3 joint audio/video DMD and TDM | Dense |
| `legacy/` | Original CogVideoX and Wan2.1 BLADE entry points | Original adaptive sparse attention |

H3 currently uses full attention. Selecting `flash_4` changes the dense attention backend; it does not enable sparse attention. Current dense recipes and original sparse paper experiments have separate entry points.

## Current training

Run current installation, preparation, training, and sampling commands from `training/`, using an environment separate from the original implementation.

```bash
cd training
# Install a matching CUDA build of PyTorch 2.9.1 first.
python -m pip install -r requirements/dense.txt
python -m pip install --no-deps -e .
# Required by the reference H3 flash_4 configuration.
python -m pip install -r requirements/h3.txt

python -m video_blade --help
bash scripts/train_h3.sh --config configs/h3_tdm.json
```

Prepare model files and prompt embeddings before training. See [current training overview](training/README.md), [H3 training](training/docs/h3-training.md), and [dense Wan2.1 training](training/docs/dense-training.md). Detailed training guides are in Chinese. The repository distributes source and example configurations, not model weights or prompt caches.

## Original implementation

For the original paper setup and sparse-attention workflows, enter `legacy/` and follow [the original English README](legacy/README.md) or [Chinese README](legacy/README_zh.md). Paths in those guides are relative to `legacy/`. The original training objectives and sparse-attention adapters are preserved; historical local paths and notebook backups have been cleaned up.

## Planned sparse-attention updates

- [x] Separate current and original implementations.
- [x] Provide dense Wan2.1 and H3 audio/video training.
- [ ] Update sparse-attention methods and kernels for the current training framework.
- [ ] Integrate and validate sparse attention for H3 training.
- [ ] Publish validated recipes and matched dense-versus-sparse quality and efficiency comparisons.

The sparse-attention items above are planned work, not features already implemented in `training/`.

## Citation

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
