from __future__ import annotations

import argparse
import shutil
import urllib.request
from pathlib import Path

from owid_dietary_common import CSV_URL, GRAPHER_PAGE_URL, TOPIC_PAGE_URL

CSV_FILENAME = "dietary-composition-by-country.csv"
GRAPHER_FILENAME = "dietary-composition-by-country.html"
TOPIC_FILENAME = "diet-compositions.html"


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Download the public OWID dietary-composition benchmark artifacts used by this repository."
    )
    parser.add_argument(
        "--output-dir",
        default=str(root_dir / "raw"),
        help="Directory where the raw OWID files will be stored.",
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

    download_file(CSV_URL, output_dir / CSV_FILENAME, force=args.force)
    download_file(GRAPHER_PAGE_URL, output_dir / GRAPHER_FILENAME, force=args.force)
    download_file(TOPIC_PAGE_URL, output_dir / TOPIC_FILENAME, force=args.force)


if __name__ == "__main__":
    main()

