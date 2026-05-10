# BLS QCEW Industry Employment Mix

Public direct-access benchmark candidate built from BLS Quarterly Census of
Employment and Wages annual singlefile CSV zips.

Default track:

- entity: state
- time: year
- vector: QCEW supersector employment shares, summing private/federal/state/local
  ownership rows by default
- output: `processed/train.pt`, `processed/val.pt`, `processed/test.pt`

```bash
python benchmarks/bls_qcew/download.py --start-year 2010 --end-year 2024
python benchmarks/bls_qcew/prepare.py --start-year 2010 --end-year 2024
```

No application or registration is required. Source files are downloaded from
`https://data.bls.gov/cew/data/files/{year}/csv/{year}_annual_singlefile.zip`.
