from pathlib import Path


FORBIDDEN_PATHS = [
    "CODE_RELEASE_PLAN.md",
    "paper_submission.zip",
    "paper" + "_writing" + "_preparation",
    "paper" + "_writing" + "_preparation" + "_internal" + "_archive",
    "s" + "batch_log",
]

FORBIDDEN_TEXT = [
    "/" + "storage" + "/home",
    "/" + "storage" + "/project",
    "paper" + "_writing" + "_preparation",
    "s" + "batch_log",
]


def test_forbidden_release_paths_are_absent():
    root = Path(__file__).resolve().parents[1]
    for rel in FORBIDDEN_PATHS:
        assert not (root / rel).exists(), rel


def test_user_facing_text_has_no_local_paths():
    root = Path(__file__).resolve().parents[1]
    text_files = [
        *root.glob("*.md"),
        *root.glob("docs/*.md"),
        *root.glob("benchmarks/**/*.md"),
        *root.glob("configs/*"),
        *root.glob("scripts/*.sh"),
    ]
    for path in text_files:
        text = path.read_text(encoding="utf-8")
        for needle in FORBIDDEN_TEXT:
            assert needle not in text, f"{needle} found in {path}"
