# Ember Monthly Electricity Benchmark

This benchmark turns official **Ember monthly electricity data** into country-level probability-vector sequences.

## Current Track

- track name: `ember_monthly_generation_mix`
- sequence unit: country / economy
- time step: month
- probability dimension: `9`
- fuel vocabulary:
  - `Coal`
  - `Gas`
  - `Other Fossil`
  - `Nuclear`
  - `Hydro`
  - `Wind`
  - `Solar`
  - `Bioenergy`
  - `Other Renewables`

The current implementation uses Ember's detailed fuel generation rows in `TWh`. These nine fuel variables sum to official `Total Generation`, so no extra residual bin is needed in the default track.

## Default Split

Each retained country sequence is split chronologically on one-step target months:

- long enough sequences: train = all but final `24` target months, val = next `12`, test = final `12`
- shorter sequences: chronological `70/15/15`

By default, the benchmark keeps the **longest contiguous monthly window** per country and requires at least `36` months in that window.

## Download

```bash
cd <repo-root>
python benchmarks/ember_monthly/download.py
```

Files written by default:

- `benchmarks/ember_monthly/raw/monthly_full_release_long_format.csv`
- `benchmarks/ember_monthly/raw/ember_electricity_data_methodology.pdf`
- `benchmarks/ember_monthly/raw/monthly-electricity-data.html`
- `benchmarks/ember_monthly/raw/api.html`

## Prepare

```bash
cd <repo-root>
python benchmarks/ember_monthly/prepare.py
python benchmarks/ember_monthly/prepare.py --min-contiguous-months 60 --split-mode tail
```

Files written by default:

- `benchmarks/ember_monthly/processed/train.pt`
- `benchmarks/ember_monthly/processed/val.pt`
- `benchmarks/ember_monthly/processed/test.pt`
- `benchmarks/ember_monthly/processed/metadata.json`
- `benchmarks/ember_monthly/processed/vocab.csv`
- `benchmarks/ember_monthly/processed/geographies.csv`

## Train

```bash
cd <repo-root>
export DATASET=ember_monthly
export MODEL=rist
export BLOCK_SIZE=24
export LOSS_TYPE=kl
torchrun --standalone --nproc_per_node=2 dist.py
```

Or use the dedicated Slurm entrypoint:

```bash
cd <repo-root>
sbatch ember_run.sbatch
```

## Split Rationale

- Keep country-level sequences because electricity-generation shares are naturally defined at that geography.
- Use Ember's detailed fuel generation rows in `TWh` and normalize them to a generation mix per month.
- Select the longest contiguous monthly window per country to avoid missing-month artifacts.
- Reserve the last `24` target months as separate validation and test windows when the country history is long enough.
- Fall back to `70/15/15` only for shorter histories so newer countries are not discarded unnecessarily.
