from __future__ import annotations

import argparse
import shutil
import urllib.request
from pathlib import Path

from cdc_weekly_common import DATASET_PAGE_URL, DOWNLOAD_CSV_URL, METADATA_JSON_URL

CSV_FILENAME = "cdc_weekly_deaths.csv"
METADATA_FILENAME = "cdc_weekly_deaths.metadata.json"
PAGE_FILENAME = "cdc_weekly_deaths.html"


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Download the public CDC weekly provisional deaths benchmark artifacts used by this repository."
    )
    parser.add_argument(
        "--output-dir",
        default=str(root_dir / "raw"),
        help="Directory where the raw CDC files will be stored.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if they already exist locally.",
    )
    return parser.parse_args()


def request(url: str) -> urllib.request.Request:
    return urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})


def download_file(url: str, output_path: Path, *, force: bool) -> None:
    if output_path.exists() and not force:
        print(f"[download] reuse {output_path}")
        return

    with urllib.request.urlopen(request(url)) as response, output_path.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    print(f"[download] wrote {output_path}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    download_file(DOWNLOAD_CSV_URL, output_dir / CSV_FILENAME, force=args.force)
    download_file(METADATA_JSON_URL, output_dir / METADATA_FILENAME, force=args.force)
    download_file(DATASET_PAGE_URL, output_dir / PAGE_FILENAME, force=args.force)


if __name__ == "__main__":
    main()

