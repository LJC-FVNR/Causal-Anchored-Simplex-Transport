from __future__ import annotations

import argparse
import shutil
import urllib.request
from pathlib import Path

DOWNLOAD_PORTAL_URL = "https://biotime.st-andrews.ac.uk/getFullDownload.php"
RAW_DATA_URL = "https://biotime.st-andrews.ac.uk/dl_request.php?dl=raw_csv"
METADATA_URL = "https://biotime.st-andrews.ac.uk/dl_request.php?dl=metadata_csv"
PORTAL_SNAPSHOT_FILENAME = "getFullDownload.html"
RAW_FILENAME = "biotime_raw.zip"
METADATA_FILENAME = "biotime_metadata.csv"


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Download the official BioTIME raw observation zip and metadata CSV."
    )
    parser.add_argument(
        "--output-dir",
        default=str(root_dir / "raw"),
        help="Directory where the raw BioTIME files will be stored.",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Only download the public metadata CSV and download-portal snapshot.",
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

    with urllib.request.urlopen(url) as response, output_path.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    print(f"[download] wrote {output_path}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    download_file(DOWNLOAD_PORTAL_URL, output_dir / PORTAL_SNAPSHOT_FILENAME, force=args.force)
    download_file(METADATA_URL, output_dir / METADATA_FILENAME, force=args.force)

    if args.metadata_only:
        print("[download] metadata-only mode enabled; skipping raw BioTIME observation download")
        return

    download_file(RAW_DATA_URL, output_dir / RAW_FILENAME, force=args.force)
    print(f"[download] portal_url={DOWNLOAD_PORTAL_URL}")
    print(f"[download] raw_data_url={RAW_DATA_URL}")
    print(f"[download] metadata_url={METADATA_URL}")


if __name__ == "__main__":
    main()
