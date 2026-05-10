from __future__ import annotations

import argparse
import shutil
import urllib.request
from pathlib import Path

MONTHLY_PAGE_URL = "https://ember-energy.org/data/data-product/monthly-electricity-data"
API_PAGE_URL = "https://ember-energy.org/data/api"
RAW_DATA_URL = "https://files.ember-energy.org/public-downloads/monthly_full_release_long_format.csv"
METHODOLOGY_URL = "https://files.ember-energy.org/public-downloads/ember_electricity_data_methodology.pdf"

RAW_FILENAME = "monthly_full_release_long_format.csv"
METHODOLOGY_FILENAME = "ember_electricity_data_methodology.pdf"
MONTHLY_PAGE_FILENAME = "monthly-electricity-data.html"
API_PAGE_FILENAME = "api.html"


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Download the official Ember monthly electricity data release and documentation."
    )
    parser.add_argument(
        "--output-dir",
        default=str(root_dir / "raw"),
        help="Directory where the raw Ember files will be stored.",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Only download the public documentation and page snapshots, not the raw CSV.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if they already exist locally.",
    )
    return parser.parse_args()


def download_file(url: str, output_path: Path, *, force: bool) -> None:
    if output_path.exists() and not force:
        print(f"[download] reuse {output_path}")
        return

    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request) as response, output_path.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    print(f"[download] wrote {output_path}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    download_file(MONTHLY_PAGE_URL, output_dir / MONTHLY_PAGE_FILENAME, force=args.force)
    download_file(API_PAGE_URL, output_dir / API_PAGE_FILENAME, force=args.force)
    download_file(METHODOLOGY_URL, output_dir / METHODOLOGY_FILENAME, force=args.force)

    if args.metadata_only:
        print("[download] metadata-only mode enabled; skipping raw Ember CSV download")
        return

    download_file(RAW_DATA_URL, output_dir / RAW_FILENAME, force=args.force)
    print(f"[download] monthly_page_url={MONTHLY_PAGE_URL}")
    print(f"[download] api_page_url={API_PAGE_URL}")
    print(f"[download] raw_data_url={RAW_DATA_URL}")
    print(f"[download] methodology_url={METHODOLOGY_URL}")


if __name__ == "__main__":
    main()
