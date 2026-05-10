# EPA AirData AQI Category Mix

Public direct-access benchmark candidate built from EPA AirData daily AQI by
county files.

Default track:

- entity: county
- time: month
- vector: ordered AQI category shares
- output: `processed/train.pt`, `processed/val.pt`, `processed/test.pt`

```bash
python benchmarks/epa_airdata_aqi/download.py --start-year 2000 --end-year 2024
python benchmarks/epa_airdata_aqi/prepare.py --start-year 2000 --end-year 2024
```

No application or registration is required for the static EPA AirData CSV/ZIP
files. The AQI category support is ordered, so the task exposes
`ordered_probability_support()`.
