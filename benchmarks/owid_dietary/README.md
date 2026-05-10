# OWID Dietary Composition Benchmark

This benchmark turns the public **Our World in Data dietary-composition grapher** into annual country-level probability-vector sequences.

## Current Track

- track name: `owid_dietary_composition_by_country`
- sequence unit: current ISO3 country entity
- time step: `year`
- probability dimension: `25`
- vocabulary:
  - `Miscellaneous`
  - `Alcoholic beverages`
  - `Animal fats`
  - `Vegetable oils`
  - `Oilcrops`
  - `Fish and seafood`
  - `Sugar crops`
  - `Sugar and sweeteners`
  - `Starchy roots`
  - `Other meat`
  - `Sheep and goat meat`
  - `Pork`
  - `Poultry meat`
  - `Beef and buffalo meat`
  - `Eggs`
  - `Milk`
  - `Nuts`
  - `Fruit`
  - `Vegetables`
  - `Pulses`
  - `Other cereals`
  - `Barley`
  - `Maize`
  - `Rice`
  - `Wheat`

The current implementation uses the OWID grapher table directly and keeps only current ISO3 entities with a complete `1961..2023` annual panel. OWID aggregate regions and historical composite entities are excluded by construction.

## Default Split

Because the retained panel is balanced over `1961..2023`, this benchmark uses a fixed chronological `70/15/15` split on one-step target years:

- train targets: `1962..2004`
- val targets: `2005..2013`
- test targets: `2014..2023`

## Download

```bash
cd <repo-root>
python benchmarks/owid_dietary/download.py
```

Files written by default:

- `benchmarks/owid_dietary/raw/dietary-composition-by-country.csv`
- `benchmarks/owid_dietary/raw/dietary-composition-by-country.html`
- `benchmarks/owid_dietary/raw/diet-compositions.html`

## Prepare

```bash
cd <repo-root>
python benchmarks/owid_dietary/prepare.py
python benchmarks/owid_dietary/prepare.py --train-fraction 0.65 --val-fraction 0.20
```

Files written by default:

- `benchmarks/owid_dietary/processed/train.pt`
- `benchmarks/owid_dietary/processed/val.pt`
- `benchmarks/owid_dietary/processed/test.pt`
- `benchmarks/owid_dietary/processed/metadata.json`
- `benchmarks/owid_dietary/processed/vocab.csv`
- `benchmarks/owid_dietary/processed/entities.csv`

The prepared tensors also retain useful sidecars:

- `total_kcal_capita_day`
- `blank_component_count`
- `observation_year`

## Train

```bash
cd <repo-root>
export DATASET=owid_dietary
export MODEL=rist
export BLOCK_SIZE=24
export LOSS_TYPE=kl
torchrun --standalone --nproc_per_node=2 dist.py
```

Or use the dedicated Slurm entrypoint:

```bash
cd <repo-root>
sbatch owid_run.sbatch
```

## Split Rationale

- Keep only current ISO3 entities with a complete `1961..2023` panel so all countries share the same benchmark window.
- Exclude OWID aggregate regions and historical composite entities to avoid mixing political units with national series.
- Normalize each country-year food-supply vector to a simplex, while retaining the original total dietary energy as a sidecar.
- Use a fixed chronological `70/15/15` split on target years so every entity is evaluated on the same held-out modern period.
