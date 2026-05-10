from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


DIST_RE = re.compile(r"num_in_sys_dist(\d+)_server1\.csv$")


def distribution_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_queue_root() -> Path:
    return Path(__file__).resolve().parent / "raw"


def default_output_dir() -> Path:
    return Path(__file__).resolve().parent / "processed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the queueing num-in-system probability benchmark for the distribution repository."
    )
    parser.add_argument(
        "--queue-root",
        default=str(default_queue_root()),
        help="Root directory of the queue benchmark raw-data bundle.",
    )
    parser.add_argument(
        "--source-variant",
        default="exploratory",
        choices=("exploratory", "homogeneous", "nonhomogeneous", "combined"),
        help="Which queue raw source(s) to migrate into the benchmark.",
    )
    parser.add_argument(
        "--source-dir",
        action="append",
        default=None,
        help="Optional explicit source directory. Can be passed multiple times; overrides --source-variant.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(default_output_dir()),
        help="Directory where processed manifests and per-sequence .npy files will be written.",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.70,
        help="Train fraction for file-level queue-system split.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.10,
        help="Validation fraction for file-level queue-system split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for deterministic split assignment.",
    )
    parser.add_argument(
        "--pad-dim",
        type=int,
        default=None,
        help="Optional explicit global probability dimension. Defaults to the maximum observed support width.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional debug cap on the number of raw CSV files to process.",
    )
    return parser.parse_args()


def source_entries(queue_root: Path, variant: str, explicit_dirs: List[str] | None) -> List[Tuple[str, Path]]:
    if explicit_dirs:
        return [
            (f"custom_{idx}", Path(path).expanduser().resolve())
            for idx, path in enumerate(explicit_dirs)
        ]

    mapping = {
        "exploratory": [
            ("exploratory", queue_root / "exploratory" / "num_in_sys_data"),
        ],
        "homogeneous": [
            (
                "homogeneous",
                queue_root / "homogeneous" / "num_in_sys_data",
            ),
        ],
        "nonhomogeneous": [
            (
                "nonhomogeneous",
                queue_root / "nonhomogeneous" / "num_in_sys_data",
            ),
        ],
        "combined": [
            (
                "homogeneous",
                queue_root / "homogeneous" / "num_in_sys_data",
            ),
            (
                "nonhomogeneous",
                queue_root / "nonhomogeneous" / "num_in_sys_data",
            ),
        ],
    }
    return mapping[variant]


def parse_dist_id(path: Path) -> int:
    match = DIST_RE.search(path.name)
    if not match:
        raise ValueError(f"could not parse dist id from filename: {path.name}")
    return int(match.group(1))


def scan_raw_csv(path: Path) -> Tuple[int, int]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        support_width = sum(1 for col in header if col.startswith("num_in_sys_"))
        length = sum(1 for _ in reader)
    if support_width <= 0:
        raise ValueError(f"{path} has no num_in_sys_* columns")
    if length <= 1:
        raise ValueError(f"{path} is too short for next-step prediction (length={length})")
    return support_width, length


def collect_records(entries: Iterable[Tuple[str, Path]], max_files: int | None) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    remaining = max_files
    for source_name, source_dir in entries:
        if not source_dir.exists():
            raise FileNotFoundError(f"missing queue source directory: {source_dir}")

        csv_files = sorted(source_dir.glob("num_in_sys_dist*_server1.csv"))
        if remaining is not None:
            csv_files = csv_files[:remaining]
        for csv_path in csv_files:
            support_width, length = scan_raw_csv(csv_path)
            rows.append(
                {
                    "source": source_name,
                    "raw_path": str(csv_path.resolve()),
                    "dist_id": parse_dist_id(csv_path),
                    "support_width": support_width,
                    "sequence_length": length,
                }
            )
            if remaining is not None:
                remaining -= 1
                if remaining <= 0:
                    break
        if remaining is not None and remaining <= 0:
            break

    if not rows:
        raise FileNotFoundError("no queue num_in_sys_dist*_server1.csv files were found")

    records = pd.DataFrame(rows).sort_values(["source", "dist_id", "raw_path"]).reset_index(drop=True)
    return records


def allocate_counts(count: int, train_fraction: float, val_fraction: float) -> Tuple[int, int, int]:
    if count <= 0:
        return 0, 0, 0
    if count == 1:
        return 1, 0, 0
    if count == 2:
        return 1, 0, 1

    fractions = np.array([train_fraction, val_fraction, 1.0 - train_fraction - val_fraction], dtype=np.float64)
    raw = fractions * count
    counts = np.floor(raw).astype(int)
    remainder = count - int(counts.sum())
    order = np.argsort(-(raw - counts))
    for idx in order[:remainder]:
        counts[idx] += 1

    for split_idx in (1, 2):
        if counts[split_idx] == 0:
            donor = int(np.argmax(counts))
            if counts[donor] > 1:
                counts[donor] -= 1
                counts[split_idx] += 1

    if counts[0] == 0:
        donor = int(np.argmax(counts[1:])) + 1
        if counts[donor] > 1:
            counts[donor] -= 1
            counts[0] += 1

    return int(counts[0]), int(counts[1]), int(counts[2])


def assign_splits(records: pd.DataFrame, train_fraction: float, val_fraction: float, seed: int) -> pd.DataFrame:
    if not (0.0 < train_fraction < 1.0):
        raise ValueError("train_fraction must lie in (0, 1)")
    if not (0.0 <= val_fraction < 1.0):
        raise ValueError("val_fraction must lie in [0, 1)")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError("train_fraction + val_fraction must be < 1")

    work = records.copy()
    rng = np.random.default_rng(seed)

    if len(work) < 10:
        order = work.index.to_numpy().copy()
        rng.shuffle(order)
        n_train, n_val, _ = allocate_counts(len(order), train_fraction=train_fraction, val_fraction=val_fraction)
        split = np.full(len(work), "test", dtype=object)
        split[order[:n_train]] = "train"
        split[order[n_train : n_train + n_val]] = "val"
        work["split"] = split
        return work

    num_bins = min(10, len(work))
    ranked = work["support_width"].rank(method="first")
    work["support_bin"] = pd.qcut(ranked, q=num_bins, labels=False, duplicates="drop")
    split = np.empty(len(work), dtype=object)

    for (_, _), group in work.groupby(["source", "support_bin"], sort=False):
        indices = group.index.to_numpy().copy()
        rng.shuffle(indices)
        n_train, n_val, _ = allocate_counts(len(indices), train_fraction=train_fraction, val_fraction=val_fraction)
        split[indices[:n_train]] = "train"
        split[indices[n_train : n_train + n_val]] = "val"
        split[indices[n_train + n_val :]] = "test"

    work["split"] = split
    if any((work["split"] == name).sum() == 0 for name in ("train", "val", "test")):
        order = work.index.to_numpy().copy()
        rng.shuffle(order)
        n_train, n_val, _ = allocate_counts(len(order), train_fraction=train_fraction, val_fraction=val_fraction)
        split = np.full(len(work), "test", dtype=object)
        split[order[:n_train]] = "train"
        split[order[n_train : n_train + n_val]] = "val"
        work["split"] = split

    return work.drop(columns=["support_bin"])


def next_source_filename(source: str, dist_id: int) -> str:
    return f"{source}__dist{dist_id:05d}.npy"


def next_time_filename(source: str, dist_id: int) -> str:
    return f"{source}__dist{dist_id:05d}.time.npy"


def numeric_prob_columns(frame: pd.DataFrame) -> List[str]:
    cols = [col for col in frame.columns if col.startswith("num_in_sys_")]
    return sorted(cols, key=lambda col: int(col.rsplit("_", 1)[1]))


def normalize_probability_matrix(values: np.ndarray) -> np.ndarray:
    values = np.clip(values.astype(np.float32, copy=False), a_min=0.0, a_max=None)
    denom = np.clip(values.sum(axis=1, keepdims=True), a_min=1e-12, a_max=None)
    return values / denom


def canonical_time_values(frame: pd.DataFrame, sequence_length: int) -> np.ndarray:
    if "time" in frame.columns:
        return frame["time"].to_numpy(dtype=np.float32)
    return np.arange(sequence_length, dtype=np.float32)


def safe_raw_relpath(raw_path: Path, raw_root: Path) -> str:
    try:
        return raw_path.resolve().relative_to(raw_root.resolve()).as_posix()
    except ValueError:
        return raw_path.name


def summarize_numeric_frame(frame: pd.DataFrame, column: str) -> Dict[str, float]:
    return {
        "min": float(frame[column].min()),
        "median": float(frame[column].median()),
        "max": float(frame[column].max()),
        "mean": float(frame[column].mean()),
    }


def prepare_split(
    output_dir: Path,
    split_name: str,
    split_records: pd.DataFrame,
    *,
    raw_root: Path,
) -> List[Dict[str, object]]:
    npy_dir = output_dir / "npy"
    time_dir = output_dir / "times"
    rows: List[Dict[str, object]] = []

    for _, record in split_records.iterrows():
        raw_path = Path(str(record["raw_path"]))
        frame = pd.read_csv(raw_path)
        prob_cols = numeric_prob_columns(frame)
        probs = normalize_probability_matrix(frame[prob_cols].to_numpy(dtype=np.float32))
        time_values = canonical_time_values(frame, sequence_length=probs.shape[0])

        npy_name = next_source_filename(str(record["source"]), int(record["dist_id"]))
        time_name = next_time_filename(str(record["source"]), int(record["dist_id"]))
        np.save(npy_dir / npy_name, probs.astype(np.float16))
        np.save(time_dir / time_name, time_values.astype(np.float32))

        sequence_row: Dict[str, object] = {
            "split": split_name,
            "source": str(record["source"]),
            "dist_id": int(record["dist_id"]),
            "raw_relpath": safe_raw_relpath(raw_path, raw_root=raw_root),
            "npy_relpath": f"npy/{npy_name}",
            "time_relpath": f"times/{time_name}",
            "sequence_length": int(probs.shape[0]),
            "support_width": int(probs.shape[1]),
            "time_start": float(time_values[0]) if len(time_values) > 0 else 0.0,
            "time_end": float(time_values[-1]) if len(time_values) > 0 else float(probs.shape[0] - 1),
        }

        if len(time_values) > 1:
            diffs = np.diff(time_values.astype(np.float64, copy=False))
            if diffs.size > 0:
                sequence_row["time_step_mean"] = float(diffs.mean())
                sequence_row["time_step_min"] = float(diffs.min())
                sequence_row["time_step_max"] = float(diffs.max())

        if "server_utilization" in frame.columns:
            util = frame["server_utilization"].to_numpy(dtype=np.float32)
            sequence_row["server_utilization_mean"] = float(util.mean())
            sequence_row["server_utilization_max"] = float(util.max())

        for col in (
            "arrival_moment_1",
            "arrival_moment_2",
            "arrival_moment_3",
            "arrival_moment_4",
            "service_moment_1",
            "service_moment_2",
            "service_moment_3",
            "service_moment_4",
        ):
            if col in frame.columns:
                sequence_row[col] = float(frame[col].iloc[0])

        rows.append(sequence_row)

    return rows


def write_vocab(output_dir: Path, prob_dim: int) -> None:
    vocab = pd.DataFrame(
        {
            "index": list(range(prob_dim)),
            "token_label": [f"n={idx}" for idx in range(prob_dim)],
        }
    )
    vocab.to_csv(output_dir / "vocab.csv", index=False)


def main() -> None:
    args = parse_args()
    queue_root = Path(args.queue_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    npy_dir = output_dir / "npy"
    time_dir = output_dir / "times"
    output_dir.mkdir(parents=True, exist_ok=True)
    npy_dir.mkdir(parents=True, exist_ok=True)
    time_dir.mkdir(parents=True, exist_ok=True)

    entries = source_entries(queue_root, args.source_variant, args.source_dir)
    records = collect_records(entries, max_files=args.max_files)
    records = assign_splits(records, train_fraction=args.train_fraction, val_fraction=args.val_fraction, seed=args.seed)

    max_support_width = int(records["support_width"].max())
    prob_dim = int(args.pad_dim or max_support_width)
    if prob_dim < max_support_width:
        raise ValueError(
            f"--pad-dim={prob_dim} is smaller than the maximum observed support width {max_support_width}"
        )

    all_sequence_rows: Dict[str, List[Dict[str, object]]] = {}
    for split_name in ("train", "val", "test"):
        split_records = records.loc[records["split"] == split_name].copy().reset_index(drop=True)
        split_rows = prepare_split(
            output_dir=output_dir,
            split_name=split_name,
            split_records=split_records,
            raw_root=queue_root,
        )
        all_sequence_rows[split_name] = split_rows
        pd.DataFrame(split_rows).to_csv(output_dir / f"{split_name}_sequences.csv", index=False)

    write_vocab(output_dir, prob_dim=prob_dim)

    metadata = {
        "benchmark": "queue_num_in_sys",
        "description": "Queueing-system occupancy-distribution benchmark with vendor-free raw bundle",
        "raw_root": str(queue_root),
        "source_variant": args.source_variant,
        "source_dirs": {name: str(path) for name, path in entries},
        "split_method": "file-level split stratified by support-width deciles within source family",
        "train_fraction": float(args.train_fraction),
        "val_fraction": float(args.val_fraction),
        "test_fraction": float(1.0 - args.train_fraction - args.val_fraction),
        "seed": int(args.seed),
        "prob_dim": prob_dim,
        "max_support_width": max_support_width,
        "num_sequences_total": int(len(records)),
        "split_counts": {
            split_name: int((records["split"] == split_name).sum())
            for split_name in ("train", "val", "test")
        },
        "support_width_stats": summarize_numeric_frame(records, "support_width"),
        "sequence_length_stats": summarize_numeric_frame(records, "sequence_length"),
        "sources": {
            source_name: int(count)
            for source_name, count in records["source"].value_counts().sort_index().items()
        },
        "ordered_support": True,
        "offline_eval_supported": False,
        "rollout_eval_supported": True,
        "time_sidecars_embedded": True,
        "files": {
            split_name: f"{split_name}_sequences.csv"
            for split_name in ("train", "val", "test")
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("[queue_prepare] wrote queue benchmark manifests to", output_dir)
    print("[queue_prepare] source variant:", args.source_variant)
    print("[queue_prepare] num sequences:", metadata["num_sequences_total"])
    print("[queue_prepare] split counts:", metadata["split_counts"])
    print("[queue_prepare] prob_dim:", metadata["prob_dim"])
    print("[queue_prepare] support width stats:", metadata["support_width_stats"])
    print("[queue_prepare] sequence length stats:", metadata["sequence_length_stats"])


if __name__ == "__main__":
    main()
