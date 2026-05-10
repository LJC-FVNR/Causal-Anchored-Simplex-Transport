#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON:-python}
TORCHRUN_BIN=${TORCHRUN:-torchrun}

"${PYTHON_BIN}" cast_smoke_check.py

DATASET=tensor_dir DATASET_DIR=tests/fixtures/tiny_simplex MODEL=cast \
ABLATION=full MAX_ITERS=2 EVAL_INTERVAL=1 EVAL_ITERS=1 BATCH_SIZE=2 \
GRAD_ACCUM_STEPS=1 D_MODEL=32 N_LAYER=1 N_HEAD=4 D_FF=64 \
USE_COMPILE=0 USE_AMP=0 RUN_OFFLINE_EVAL=0 RUN_ROLLOUT_EVAL=0 \
SAVE_BEST_CHECKPOINT=0 SKIP_COMPLETED_IDENTICAL=0 \
"${TORCHRUN_BIN}" --standalone --nproc_per_node=1 dist.py
