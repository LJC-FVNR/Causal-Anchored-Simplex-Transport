from __future__ import annotations

import argparse
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simplex_benchmark_utils import download_file, month_sequence

TRIP_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/{service}_tripdata_{month}.parquet"
ZONE_LOOKUP_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
TLC_PAGE_URL = "https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page"


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Download NYC TLC trip-record Parquet files.")
    parser.add_argument("--output-dir", default=str(root_dir / "raw"))
    parser.add_argument("--service", default="yellow", choices=("yellow", "green", "fhv", "fhvhv"))
    parser.add_argument("--start-month", default="2023-01")
    parser.add_argument("--end-month", default="2023-12")
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    download_file(TLC_PAGE_URL, output_dir / "tlc-trip-record-data.html", force=args.force)
    download_file(ZONE_LOOKUP_URL, output_dir / "taxi_zone_lookup.csv", force=args.force)

    if args.metadata_only:
        print("[download] metadata-only mode enabled; skipping TLC Parquet trip files")
        return

    for month in month_sequence(args.start_month, args.end_month):
        url = TRIP_URL.format(service=args.service, month=month)
        download_file(url, output_dir / f"{args.service}_tripdata_{month}.parquet", force=args.force)
    print(f"[download] nyc_tlc service={args.service} months={args.start_month}..{args.end_month}")


if __name__ == "__main__":
    main()
