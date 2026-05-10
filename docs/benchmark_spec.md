# Benchmark Spec

The paper reports 11 benchmark sections:

1. BioTIME
2. Ember Monthly
3. OWID Dietary
4. CDC Weekly Deaths
5. BLS QCEW
6. EPA AirData AQI
7. NOAA Storm Events
8. NYC TLC Trip
9. Queue Homogeneous
10. Queue Nonhomogeneous
11. Queue Combined

The eight public-source tensor benchmarks are prepared with:

```bash
python benchmarks/<dataset>/download.py
python benchmarks/<dataset>/prepare.py
```

Prepared tensor datasets are loaded with `DATASET=<dataset>` or directly with
`DATASET=tensor_dir DATASET_DIR=<processed-dir>`.

Queue sections use whole-system file-level splits, stratified by support-width
deciles with split seed 42. For exact paper reproduction, use the processed
queue artifacts:

```text
benchmarks/queue_num_in_sys/processed_variants/homogeneous
benchmarks/queue_num_in_sys/processed_variants/nonhomogeneous
benchmarks/queue_num_in_sys/processed_variants/combined
```

The main one-step metric is test offline KL. The main rollout metric is test
rollout JSD. Additional reported metrics include KL, JSD, L1, Bray-Curtis, and
ordered-support Wasserstein-1 after simplex normalization with epsilon `1e-8`.
