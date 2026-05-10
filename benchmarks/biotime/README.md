# BioTIME Benchmark

This benchmark turns public **BioTIME 2.0** community observations into annual probability-vector sequences.

## Current Track

- track name: `biotime_birds_species_annual`
- taxon subset: `Birds`
- sequence unit: `study_id x latitude x longitude x depth`
- time step: `year`
- vocabulary: global top-`127` bird species by retained measure mass, plus `OTHER`
- probability dimension: `128`

The current implementation keeps only `resolution=species` rows and chooses the **longest contiguous annual window** per site. By default, sites need at least `15` annual observations in that longest window.

## Default Split

Each retained site sequence is split chronologically on one-step target years:

- train: first `70%`
- val: next `15%`
- test: final `15%`

This happens per sequence after the longest contiguous annual window is selected.

## Download

```bash
cd <repo-root>
python benchmarks/biotime/download.py
```

Files written by default:

- `benchmarks/biotime/raw/biotime_raw.zip`
- `benchmarks/biotime/raw/biotime_metadata.csv`
- `benchmarks/biotime/raw/getFullDownload.html`

The downloader uses the official BioTIME portal:

- https://biotime.st-andrews.ac.uk/getFullDownload.php

## Prepare

```bash
cd <repo-root>
python benchmarks/biotime/prepare.py
python benchmarks/biotime/prepare.py --taxon Birds --top-k 255 --min-contiguous-years 20
```

Files written by default:

- `benchmarks/biotime/processed/train.pt`
- `benchmarks/biotime/processed/val.pt`
- `benchmarks/biotime/processed/test.pt`
- `benchmarks/biotime/processed/metadata.json`
- `benchmarks/biotime/processed/vocab.csv`
- `benchmarks/biotime/processed/sites.csv`

## Train

```bash
cd <repo-root>
export DATASET=biotime
export MODEL=rist
export BLOCK_SIZE=8
export LOSS_TYPE=kl
torchrun --standalone --nproc_per_node=2 dist.py
```

Or use the dedicated Slurm entrypoint:

```bash
cd <repo-root>
sbatch biotime_run.sbatch
```

## Split Rationale

- Restrict to `Birds` for the current unified track because a cross-taxon global species vocabulary would be ecologically incoherent.
- Keep only species-level rows so each probability vector represents a single taxonomic resolution.
- Select the longest contiguous annual window per site to avoid irregular-time artifacts in a model that currently assumes regular steps.
- Require at least `15` annual observations so each sequence has a non-trivial train / val / test split.
- Use a global top-`127` species vocabulary plus `OTHER` to keep the benchmark trainable while preserving most retained measure mass.
