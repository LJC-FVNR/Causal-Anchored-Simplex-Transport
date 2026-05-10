#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <dataset> [model]" >&2
  exit 2
fi

dataset=$1
model=${2:-cast}

set -a
if [[ "$dataset" == queue_* || "$dataset" == "queue_num_in_sys" ]]; then
  source configs/paper_queue.env
else
  source configs/paper_nonqueue.env
fi
set +a

DATASET="$dataset" MODEL="$model" torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-1}" dist.py
