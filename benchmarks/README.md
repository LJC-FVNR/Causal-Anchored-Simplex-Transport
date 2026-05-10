# Benchmarks

Each public benchmark directory contains a downloader, preprocessing script,
and dataset-specific notes.

Typical usage:

```bash
python benchmarks/<dataset>/download.py
python benchmarks/<dataset>/prepare.py
```

The release repository does not commit large `raw/` or `processed/` data
directories. Use the scripts to regenerate them or fetch the separate processed
artifact bundle.

Queue data are handled separately because exact paper reproduction uses
project-generated processed trajectory artifacts. See
`benchmarks/queue_num_in_sys/README.md` and `docs/data_manifest.md`.
