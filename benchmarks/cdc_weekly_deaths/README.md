# CDC Weekly Deaths Benchmark

This benchmark turns the public **CDC Weekly Provisional Counts of Deaths by State and Select Causes** table into weekly state-level probability-vector sequences.

## Current Track

- track name: `cdc_weekly_state_cause_share`
- sequence unit: U.S. state
- time step: `week`
- probability dimension: `6`
- cause vocabulary:
  - `Diseases of heart`
  - `Malignant neoplasms`
  - `Cerebrovascular diseases`
  - `Chronic lower respiratory diseases`
  - `COVID-19 underlying`
  - `OTHER_CAUSES`

The current implementation aggregates `New York` and `New York City` into a single full-state New York sequence, excludes `United States`, `Puerto Rico`, and `District of Columbia`, and imputes suppressed retained-cause counts `1-9` to a midpoint of `5.0`.

## Default Filters And Split

- start from the frozen CDC `2020..2023` weekly snapshot
- drop the final `4` reported weeks of the frozen snapshot to reduce end-of-series under-reporting
- retain the resulting balanced `50`-state panel
- apply a tail holdout of:
  - validation target weeks: final `52` weeks before test
  - test target weeks: final `52` retained weeks

With the current public snapshot this yields:

- retained observation span: `2020W01 .. 2023W33`
- train targets: `2020W02 .. 2021W33`
- val targets: `2021W34 .. 2022W33`
- test targets: `2022W34 .. 2023W33`

## Download

```bash
cd <repo-root>
python benchmarks/cdc_weekly_deaths/download.py
```

Files written by default:

- `benchmarks/cdc_weekly_deaths/raw/cdc_weekly_deaths.csv`
- `benchmarks/cdc_weekly_deaths/raw/cdc_weekly_deaths.metadata.json`
- `benchmarks/cdc_weekly_deaths/raw/cdc_weekly_deaths.html`

## Prepare

```bash
cd <repo-root>
python benchmarks/cdc_weekly_deaths/prepare.py
python benchmarks/cdc_weekly_deaths/prepare.py --tail-drop-weeks 0 --split-mode tail
```

Files written by default:

- `benchmarks/cdc_weekly_deaths/processed/train.pt`
- `benchmarks/cdc_weekly_deaths/processed/val.pt`
- `benchmarks/cdc_weekly_deaths/processed/test.pt`
- `benchmarks/cdc_weekly_deaths/processed/metadata.json`
- `benchmarks/cdc_weekly_deaths/processed/vocab.csv`
- `benchmarks/cdc_weekly_deaths/processed/states.csv`
- `benchmarks/cdc_weekly_deaths/processed/weeks.csv`

The prepared tensors also retain useful sidecars:

- `total_weekly_deaths`
- `other_causes_count`
- `suppressed_selected_count`
- `mmwr_year`
- `mmwr_week`

## Train

```bash
cd <repo-root>
export DATASET=cdc_weekly_deaths
export MODEL=rist
export BLOCK_SIZE=52
export LOSS_TYPE=kl
torchrun --standalone --nproc_per_node=2 dist.py
```

Or use the dedicated Slurm entrypoint:

```bash
cd <repo-root>
sbatch cdc_run.sbatch
```

## Split Rationale

- Use the frozen CDC weekly state table because it is public and no-login.
- Aggregate `New York` and `New York City` so the benchmark uses full-state New York rather than split jurisdictions.
- Keep a compact set of low-suppression causes plus `OTHER_CAUSES` so the cause-share vectors remain interpretable and stable.
- Drop the final few reported weeks from the frozen provisional snapshot because the tail shows clear end-of-series under-reporting.
- Use a `52 + 52` week tail holdout to test full-season generalization across states.
