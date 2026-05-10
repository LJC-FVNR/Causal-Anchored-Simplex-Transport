# Data Manifest

This repository does not commit large raw or processed benchmark data. The code
tree contains download and preprocessing scripts plus small test fixtures.

Recommended external artifact layout:

```text
cast-code-release-artifacts/
├── public_processed/
├── queue_processed_variants/
├── results_metrics/
├── results_figures/
└── MANIFEST.sha256
```

## Public Tensor Benchmarks

| Dataset | Code | Redistribution note |
|---|---|---|
| BioTIME | `benchmarks/biotime/` | cite BioTIME and preserve study metadata terms |
| Ember Monthly | `benchmarks/ember_monthly/` | preserve Ember source attribution |
| OWID Dietary | `benchmarks/owid_dietary/` | preserve OWID and upstream source attribution |
| CDC Weekly Deaths | `benchmarks/cdc_weekly_deaths/` | public U.S. government source |
| BLS QCEW | `benchmarks/bls_qcew/` | public BLS source |
| EPA AirData AQI | `benchmarks/epa_airdata_aqi/` | public EPA source |
| NOAA Storm Events | `benchmarks/noaa_storm_events/` | public NOAA source |
| NYC TLC Trip | `benchmarks/nyc_tlc_trip/` | NYC Open Data terms |

For each processed artifact, include:

- generation command;
- code commit or release tag;
- file size;
- SHA256 hash;
- upstream source URL and citation;
- any license or terms-of-use note.

## Queue Artifacts

Queue data are project-generated simulation artifacts following the protocol
described in the paper. Exact paper reproduction uses:

| Section | Path inside artifact bundle |
|---|---|
| Queue Homogeneous | `queue_processed_variants/homogeneous/` |
| Queue Nonhomogeneous | `queue_processed_variants/nonhomogeneous/` |
| Queue Combined | `queue_processed_variants/combined/` |

Each queue slice should include `train_sequences.csv`, `val_sequences.csv`,
`test_sequences.csv`, `metadata.json`, `vocab.csv`, `npy/*.npy`, and
`times/*.npy`, plus a checksum manifest.
