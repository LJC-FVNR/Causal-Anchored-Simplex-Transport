# NYC TLC Pickup-Zone Mix

Public direct-access benchmark candidate built from NYC TLC monthly trip-record
Parquet files.

Default track:

- service: yellow taxi
- entity: pickup hour (`hour_00` ... `hour_23`)
- time: week
- vector: pickup taxi-zone shares plus `UNKNOWN`
- output: `processed/train.pt`, `processed/val.pt`, `processed/test.pt`

```bash
python benchmarks/nyc_tlc_trip/download.py --service yellow --start-month 2023-01 --end-month 2023-12
python benchmarks/nyc_tlc_trip/prepare.py --service yellow --start-month 2023-01 --end-month 2023-12
```

No application or registration is required. Processing requires an environment
with `pandas` and `pyarrow`.
