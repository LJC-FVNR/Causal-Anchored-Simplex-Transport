import subprocess
import sys
from pathlib import Path


def test_benchmark_scripts_show_help():
    root = Path(__file__).resolve().parents[1]
    scripts = sorted((root / "benchmarks").glob("*/download.py"))
    scripts += sorted((root / "benchmarks").glob("*/prepare.py"))
    assert scripts
    for script in scripts:
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert result.returncode == 0, f"{script} --help failed: {result.stderr}"


def test_queue_review_artifact_downloader_shows_help():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "download_queue_review_artifact.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert result.returncode == 0, result.stderr
