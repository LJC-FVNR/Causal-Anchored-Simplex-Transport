# CAST: Causal Anchored Simplex Transport

This repository contains the code release for **CAST: Causal Anchored Simplex
Transport for Distribution-Valued Time Series**.

CAST forecasts sequences whose observations are probability vectors. The model
retrieves causal empirical successors, mixes them with a persistence anchor,
and applies bounded local transport on ordered supports while preserving the
simplex.

## Contents

- `models/cast.py`, `models/cast_core.py`, `models/cast_shared.py`: CAST model.
- `training/`, `dist.py`: training and evaluation entry points.
- `tasks/`: task adapters for tensor datasets and the paper benchmarks.
- `benchmarks/`: public benchmark download and preprocessing scripts.
- `evaluation/`: offline, rollout, and visualization utilities.
- `configs/`: paper run templates and rollout settings.
- `results/`: submitted-result tables and figures used for traceability.
- `tests/`: CPU smoke and release checks.

Large raw and processed benchmark data are not committed directly. Use the
download/preparation scripts or the separately staged artifact bundle described
in `docs/data_manifest.md`.

## Install

```bash
conda env create -f environment.yml
conda activate cast
python cast_smoke_check.py
pytest -q
```

For pip-only environments:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python cast_smoke_check.py
```

## Quick Smoke Run

```bash
bash scripts/run_smoke.sh
```

The smoke script runs a tiny CPU CAST configuration on
`tests/fixtures/tiny_simplex/` and writes outputs under `outputs/`.

## Model Names

Use:

```bash
MODEL=cast ABLATION=full
```

Public CAST ablations:

- `full`: paper CAST model with structural transport regularization.
- `without_structural_reg`
- `anchor_only`
- `single_head_retrieval`
- `fixed_local_kernel`
- `without_persistence_mix`

Baselines include `persistence`, `seasonal_naive`, `analog_successor`,
`bdarma`, `bdarma_map`, `compositional_ets`, `ilr_var`,
`transformer_naive`, `dlinear`, `timemixer`, `itransformer`, `informer`,
`autoformer`, `tide`, `tcn`, `gru`, `lstm`, and `nhits`.

## Prepare Benchmarks

Each public benchmark has a downloader and preparation script:

```bash
python benchmarks/owid_dietary/download.py
python benchmarks/owid_dietary/prepare.py
```

To prepare all public-source benchmarks:

```bash
bash scripts/prepare_public_benchmarks.sh
```

The queue benchmark uses processed trajectory artifacts by default for exact
paper reproduction. Point `QUEUE_NUM_IN_SYS_DIR` to one of:

```text
benchmarks/queue_num_in_sys/processed_variants/homogeneous
benchmarks/queue_num_in_sys/processed_variants/nonhomogeneous
benchmarks/queue_num_in_sys/processed_variants/combined
```

For review-time reproduction, the queue artifact is available through the
anonymous Google Drive link documented in `docs/data_manifest.md`. Download,
verify, and unpack it with:

```bash
python scripts/download_queue_review_artifact.py --extract
```

## Train CAST

Tiny CPU example:

```bash
DATASET=tensor_dir DATASET_DIR=tests/fixtures/tiny_simplex MODEL=cast \
ABLATION=full MAX_ITERS=2 EVAL_INTERVAL=1 EVAL_ITERS=1 BATCH_SIZE=2 \
GRAD_ACCUM_STEPS=1 D_MODEL=32 N_LAYER=1 N_HEAD=4 D_FF=64 \
USE_AMP=0 USE_COMPILE=0 RUN_OFFLINE_EVAL=0 RUN_ROLLOUT_EVAL=0 \
torchrun --standalone --nproc_per_node=1 dist.py
```

Paper-style runs are documented in `docs/reproducing_results.md` and templated
in `configs/paper_nonqueue.env` and `configs/paper_queue.env`.

## Results

Submitted paper tables and figures are available under:

- `results/tables/`
- `results/figures/`

`results/README.md` describes which files are static submitted artifacts and
which files are expected from fresh runs.

## License And Citation

Code is released under the license in `LICENSE`. Data artifacts retain their
upstream terms; see `docs/data_manifest.md`.

For citation metadata, see `CITATION.cff`.
