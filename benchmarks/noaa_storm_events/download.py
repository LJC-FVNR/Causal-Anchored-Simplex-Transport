from __future__ import annotations

import argparse
import re
import urllib.request
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simplex_benchmark_utils import download_file

CSV_DIR_URL = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Download NOAA Storm Events details CSV gzip files.")
    parser.add_argument("--output-dir", default=str(root_dir / "raw"))
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def fetch_listing() -> str:
    request = urllib.request.Request(CSV_DIR_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request) as response:
        return response.read().decode("utf-8", errors="replace")


def latest_details_filename(listing: str, year: int) -> str:
    pattern = rf"StormEvents_details-ftp_v1\.0_d{year}_c\d+\.csv\.gz"
    matches = sorted(set(re.findall(pattern, listing)))
    if not matches:
        raise FileNotFoundError(f"could not find NOAA StormEvents details file for year={year}")
    return matches[-1]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    listing = fetch_listing()
    (output_dir / "csvfiles_index.html").write_text(listing, encoding="utf-8")
    print(f"[download] wrote {output_dir / 'csvfiles_index.html'}")

    if args.metadata_only:
        print("[download] metadata-only mode enabled; skipping yearly StormEvents files")
        return

    for year in range(args.start_year, args.end_year + 1):
        filename = latest_details_filename(listing, year)
        download_file(CSV_DIR_URL + filename, output_dir / filename, force=args.force)
    print(f"[download] noaa_storm_events_years={args.start_year}..{args.end_year}")


if __name__ == "__main__":
    main()
