from __future__ import annotations

import argparse
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simplex_benchmark_utils import download_file

BASE_URL = "https://data.bls.gov/cew/data/files/{year}/csv/{year}_annual_singlefile.zip"
AREA_TITLES_URL = "https://data.bls.gov/cew/doc/titles/area/area_titles.csv"
INDUSTRY_TITLES_URL = "https://data.bls.gov/cew/doc/titles/industry/industry_titles.csv"


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Download public BLS QCEW annual singlefile CSV zips.")
    parser.add_argument("--output-dir", default=str(root_dir / "raw"))
    parser.add_argument("--start-year", type=int, default=2010)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    download_file(AREA_TITLES_URL, output_dir / "area_titles.csv", force=args.force)
    download_file(INDUSTRY_TITLES_URL, output_dir / "industry_titles.csv", force=args.force)

    if args.metadata_only:
        print("[download] metadata-only mode enabled; skipping annual QCEW zips")
        return

    for year in range(args.start_year, args.end_year + 1):
        url = BASE_URL.format(year=year)
        download_file(url, output_dir / f"{year}_annual_singlefile.zip", force=args.force)
    print(f"[download] qcew_years={args.start_year}..{args.end_year}")


if __name__ == "__main__":
    main()
