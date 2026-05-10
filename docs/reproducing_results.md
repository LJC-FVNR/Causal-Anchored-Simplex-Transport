# Reproducing Results

## Smoke Test

```bash
bash scripts/run_smoke.sh
```

## One Public Dataset

```bash
python benchmarks/owid_dietary/download.py
python benchmarks/owid_dietary/prepare.py
DATASET=owid_dietary MODEL=cast ABLATION=full \
MAX_ITERS=1000 EVAL_INTERVAL=100 EVAL_ITERS=20 \
USE_COMPILE=0 torchrun --standalone --nproc_per_node=1 dist.py
```

## Paper-Style Non-Queue Run

```bash
set -a
source configs/paper_nonqueue.env
set +a
DATASET=owid_dietary torchrun --standalone --nproc_per_node=1 dist.py
```

## Paper-Style Queue Run

```bash
set -a
source configs/paper_queue.env
set +a
DATASET=queue_num_in_sys \
QUEUE_NUM_IN_SYS_DIR=benchmarks/queue_num_in_sys/processed_variants/homogeneous \
torchrun --standalone --nproc_per_node=1 dist.py
```

Repeat the queue run with `nonhomogeneous` and `combined` for the other queue
sections.

## Outputs

Fresh runs write by default to:

- `outputs/checkpoints/`
- `outputs/eval_artifacts/`
- `outputs/rollout_artifacts/`

Submitted static paper tables and figures are under `results/`.
