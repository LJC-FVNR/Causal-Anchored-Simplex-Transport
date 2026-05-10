# NOAA Storm Events Event-Type Mix

Public direct-access benchmark candidate built from NOAA Storm Events yearly
details CSV gzip files.

Default track:

- entity: U.S. state
- time: month
- vector: top event-type shares plus `OTHER`
- output: `processed/train.pt`, `processed/val.pt`, `processed/test.pt`

```bash
python benchmarks/noaa_storm_events/download.py --start-year 2000 --end-year 2024
python benchmarks/noaa_storm_events/prepare.py --start-year 2000 --end-year 2024
```

No application or registration is required. The downloader reads the official
NOAA directory and chooses the latest correction file for each requested year.
