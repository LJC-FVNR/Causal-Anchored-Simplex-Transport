# Troubleshooting

## `ModuleNotFoundError: torch`

Install dependencies with either `environment.yml` or `requirements.txt`.

## CUDA Is Not Available

Smoke tests run on CPU. Use `USE_AMP=0 USE_COMPILE=0` for portable debugging.

## Dataset Directory Missing

For `DATASET=tensor_dir`, set `DATASET_DIR` to a directory containing
`train.pt` and `val.pt`.

## Queue Artifacts Missing

Set `QUEUE_NUM_IN_SYS_DIR` to one processed queue slice. Exact paper slices are
`homogeneous`, `nonhomogeneous`, and `combined`.

For review-time reproduction, run:

```bash
python scripts/download_queue_review_artifact.py --extract
```

## NYC TLC Preparation Fails

Install `pyarrow`; the TLC pipeline reads parquet files.

## Distributed Launch Fails

For a single-process run, use:

```bash
torchrun --standalone --nproc_per_node=1 dist.py
```
