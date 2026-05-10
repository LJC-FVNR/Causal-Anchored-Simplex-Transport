# Queue Num-In-System Benchmark

This benchmark contains queue occupancy distributions over states
`N_t = 0, 1, ...`. Each sequence is one queue-system configuration, and each
time step is a simplex-valued occupancy distribution.

The paper reports three queue sections:

| Section | Processed artifact path |
|---|---|
| Queue Homogeneous | `benchmarks/queue_num_in_sys/processed_variants/homogeneous/` |
| Queue Nonhomogeneous | `benchmarks/queue_num_in_sys/processed_variants/nonhomogeneous/` |
| Queue Combined | `benchmarks/queue_num_in_sys/processed_variants/combined/` |

These processed artifacts are distributed separately from the git repository.
After placing an artifact under one of the paths above, run with:

```bash
DATASET=queue_num_in_sys \
QUEUE_NUM_IN_SYS_DIR=benchmarks/queue_num_in_sys/processed_variants/homogeneous \
MODEL=cast ABLATION=full torchrun --standalone --nproc_per_node=1 dist.py
```

## Split Policy

Splits are by whole queue system, not overlapping windows, to avoid leakage
between train and test windows from the same simulated configuration.

- split fractions: 70 / 10 / 20
- split seed: 42
- stratification: support-width deciles inside each source family

## Processed Layout

Each processed slice contains:

- `metadata.json`
- `vocab.csv`
- `train_sequences.csv`
- `val_sequences.csv`
- `test_sequences.csv`
- `npy/*.npy`
- `times/*.npy`

The `.npy` files keep each system in its natural shape. Padding to the global
dimension happens lazily during batch construction.

## Evaluation

The queue task supports deterministic streamed one-step offline evaluation via
fixed-window payloads controlled by:

- `QUEUE_OFFLINE_NUM_SEQUENCES`
- `QUEUE_OFFLINE_WINDOW_LENGTH`

It also supports autoregressive rollout evaluation via:

- `ROLLOUT_CONTEXT_STEPS`
- `ROLLOUT_HORIZON_STEPS`
- `ROLLOUT_MAX_EXAMPLES`

Because queue occupancy states are ordered, evaluation can include
Wasserstein-1 in addition to KL, JSD, L1, and Bray-Curtis.
