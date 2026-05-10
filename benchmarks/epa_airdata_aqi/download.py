from __future__ import annotations

import argparse
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simplex_benchmark_utils import download_file

BASE_URL = "https://aqs.epa.gov/aqsweb/airdata/daily_aqi_by_county_{year}.zip"
DOWNLOAD_PAGE_URL = "https://aqs.epa.gov/aqsweb/airdata/download_files.html"


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Download EPA AirData daily AQI by county public CSV zips.")
    parser.add_argument("--output-dir", default=str(root_dir / "raw"))
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    download_file(DOWNLOAD_PAGE_URL, output_dir / "download_files.html", force=args.force)
    if args.metadata_only:
        print("[download] metadata-only mode enabled; skipping AQI zips")
        return

    for year in range(args.start_year, args.end_year + 1):
        url = BASE_URL.format(year=year)
        download_file(url, output_dir / f"daily_aqi_by_county_{year}.zip", force=args.force)
    print(f"[download] epa_airdata_years={args.start_year}..{args.end_year}")


if __name__ == "__main__":
    main()
