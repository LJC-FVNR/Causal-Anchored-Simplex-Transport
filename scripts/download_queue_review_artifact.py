#!/usr/bin/env python3
"""Download the review-time queue benchmark artifact from Google Drive."""

from __future__ import annotations

import argparse
import hashlib
import html
import os
import re
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from http.cookiejar import CookieJar
from pathlib import Path


DEFAULT_DRIVE_URL = "https://drive.google.com/file/d/1-4cahBtZOigTvfEWYfCuYD51NDDSEHzG/view?usp=sharing"
DEFAULT_FILE_ID = "1-4cahBtZOigTvfEWYfCuYD51NDDSEHzG"
DEFAULT_ARCHIVE_NAME = "queue_benchmark_review_artifact_2026-05-09.tar.zst"
DEFAULT_SHA256 = "ea171e6ed6ca6829cc481e20b496d3459f05b47501826e0ac3e51fa25749ab2f"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the queue benchmark review artifact, verify its SHA256 "
            "checksum, and optionally extract it into benchmarks/queue_num_in_sys/."
        )
    )
    parser.add_argument("--url", default=DEFAULT_DRIVE_URL, help="Google Drive sharing URL.")
    parser.add_argument("--file-id", default=None, help="Google Drive file id. Overrides --url parsing.")
    parser.add_argument(
        "--archive",
        default=str(Path("benchmarks") / "queue_num_in_sys" / DEFAULT_ARCHIVE_NAME),
        help="Local archive path to write.",
    )
    parser.add_argument(
        "--sha256",
        default=DEFAULT_SHA256,
        help="Expected SHA256 checksum for the archive. Use an empty string to skip.",
    )
    parser.add_argument("--force", action="store_true", help="Re-download and re-extract existing files.")
    parser.add_argument("--skip-download", action="store_true", help="Verify/extract an existing local archive.")
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract the archive and create benchmarks/queue_num_in_sys/processed_variants.",
    )
    parser.add_argument(
        "--extract-root",
        default=str(Path("benchmarks") / "queue_num_in_sys"),
        help="Directory where the archive root should be extracted.",
    )
    return parser.parse_args()


def drive_file_id(url: str, explicit_file_id: str | None) -> str:
    if explicit_file_id:
        return explicit_file_id
    parsed = urllib.parse.urlparse(url)
    query_id = urllib.parse.parse_qs(parsed.query).get("id")
    if query_id:
        return query_id[0]
    match = re.search(r"/file/d/([^/]+)", parsed.path)
    if match:
        return match.group(1)
    if re.fullmatch(r"[-_A-Za-z0-9]+", url):
        return url
    raise ValueError(f"could not parse Google Drive file id from: {url}")


def is_file_response(response: urllib.response.addinfourl) -> bool:
    content_type = response.headers.get("Content-Type", "").lower()
    disposition = response.headers.get("Content-Disposition", "").lower()
    return "attachment" in disposition or "text/html" not in content_type


def confirm_url_from_html(page: str, base_url: str) -> str | None:
    form_match = re.search(
        r'<form[^>]+id="download-form"[^>]+action="([^"]+)"[^>]*>(.*?)</form>',
        page,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if form_match:
        action = html.unescape(form_match.group(1))
        fields = dict(
            (html.unescape(name), html.unescape(value))
            for name, value in re.findall(
                r'<input[^>]+name="([^"]+)"[^>]+value="([^"]*)"',
                form_match.group(2),
                flags=re.IGNORECASE,
            )
        )
        return urllib.parse.urljoin(base_url, action) + "?" + urllib.parse.urlencode(fields)

    href_match = re.search(r'href="([^"]*(?:confirm=|uc\?export=download)[^"]*)"', page)
    if href_match:
        return urllib.parse.urljoin(base_url, html.unescape(href_match.group(1)))

    return None


def open_drive_download(file_id: str) -> urllib.response.addinfourl:
    cookie_jar = CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))
    base_url = "https://drive.google.com/uc?" + urllib.parse.urlencode(
        {"export": "download", "id": file_id}
    )
    request = urllib.request.Request(base_url, headers={"User-Agent": "CAST-data-release/1.0"})
    response = opener.open(request)
    if is_file_response(response):
        return response

    page = response.read().decode("utf-8", errors="replace")
    confirm = next((cookie.value for cookie in cookie_jar if cookie.name.startswith("download_warning")), None)
    if confirm:
        confirm_url = base_url + "&" + urllib.parse.urlencode({"confirm": confirm})
    else:
        confirm_url = confirm_url_from_html(page, base_url)
    if not confirm_url:
        raise RuntimeError("Google Drive returned an HTML page, but no confirmation link was found.")

    request = urllib.request.Request(confirm_url, headers={"User-Agent": "CAST-data-release/1.0"})
    response = opener.open(request)
    if not is_file_response(response):
        raise RuntimeError("Google Drive did not return a file after confirmation.")
    return response


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path: Path, expected: str) -> None:
    if not expected:
        return
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"SHA256 mismatch for {path}: expected {expected}, got {actual}")
    print(f"[queue-artifact] verified sha256: {actual}")


def download_archive(file_id: str, output_path: Path, expected_sha256: str, force: bool) -> None:
    if output_path.exists() and not force:
        verify_sha256(output_path, expected_sha256)
        print(f"[queue-artifact] reuse {output_path}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    print(f"[queue-artifact] downloading Google Drive file {file_id}")
    response = open_drive_download(file_id)
    written = 0
    with tmp_path.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            written += len(chunk)
            if written and written % (256 * 1024 * 1024) < len(chunk):
                print(f"[queue-artifact] downloaded {written / 1024 / 1024:.0f} MiB")

    verify_sha256(tmp_path, expected_sha256)
    tmp_path.replace(output_path)
    print(f"[queue-artifact] wrote {output_path}")


def extract_archive(archive_path: Path, extract_root: Path, force: bool) -> None:
    tar = shutil.which("tar")
    unzstd = shutil.which("unzstd") or shutil.which("zstd")
    if not tar or not unzstd:
        raise RuntimeError("Extraction requires `tar` and `unzstd` or `zstd` on PATH.")

    artifact_root = extract_root / DEFAULT_ARCHIVE_NAME.removesuffix(".tar.zst")
    variants_link = extract_root / "processed_variants"
    if artifact_root.exists() and force:
        shutil.rmtree(artifact_root)
    if variants_link.exists() or variants_link.is_symlink():
        if not force:
            print(f"[queue-artifact] processed variants already exist: {variants_link}")
            return
        if variants_link.is_symlink():
            variants_link.unlink()
        else:
            raise RuntimeError(f"refusing to remove non-symlink directory: {variants_link}")

    extract_root.mkdir(parents=True, exist_ok=True)
    program = "unzstd" if Path(unzstd).name == "unzstd" else "zstd -d"
    subprocess.run(
        [tar, "--use-compress-program", program, "-xf", os.fspath(archive_path), "-C", os.fspath(extract_root)],
        check=True,
    )
    target = artifact_root / "queue_processed_variants"
    if not target.exists():
        raise RuntimeError(f"extracted artifact is missing {target}")
    variants_link.symlink_to(target.relative_to(extract_root), target_is_directory=True)
    print(f"[queue-artifact] linked {variants_link} -> {target}")


def main() -> int:
    args = parse_args()
    archive_path = Path(args.archive)
    file_id = drive_file_id(args.url, args.file_id)

    try:
        if not args.skip_download:
            download_archive(file_id, archive_path, args.sha256, args.force)
        else:
            verify_sha256(archive_path, args.sha256)
        if args.extract:
            extract_archive(archive_path, Path(args.extract_root), args.force)
    except Exception as exc:  # pragma: no cover - command-line reporting
        print(f"[queue-artifact] error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
