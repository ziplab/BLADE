#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_ROOT"
TASK_PYTHON="${VIDEO_BLADE_PYTHON:-python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
exec "$TASK_PYTHON" -m torch.distributed.run --standalone --nnodes=1 \
  --nproc_per_node="${NPROC_PER_NODE:-8}" -m video_blade.h3.train "$@"
