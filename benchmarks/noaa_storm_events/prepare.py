from __future__ import annotations

import argparse
import csv
import gzip
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List

import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simplex_benchmark_utils import (
    infer_split_masks,
    longest_contiguous_window,
    mask_range,
    month_sequence,
    month_to_index,
    index_to_month,
    pad_sequences,
    save_tensor_splits,
    summarize_lengths,
    write_csv,
    write_json,
)

TRACK_NAME = "noaa_storm_events_state_monthly_event_type_mix"
DETAILS_RE = re.compile(r"StormEvents_details-ftp_v1\.0_d(?P<year>\d{4})_c\d+\.csv\.gz")


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parent
    raw_dir = root_dir / "raw"
    parser = argparse.ArgumentParser(description="Prepare a NOAA Storm Events monthly event-type benchmark.")
    parser.add_argument("--input-dir", default=str(raw_dir))
    parser.add_argument("--output-dir", default=str(root_dir / "processed"))
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--entity-level", choices=("state",), default="state")
    parser.add_argument("--top-k-event-types", type=int, default=31)
    parser.add_argument("--min-contiguous-months", type=int, default=60)
    parser.add_argument("--split-mode", choices=("auto", "tail", "fraction"), default="auto")
    parser.add_argument("--val-months", type=int, default=12)
    parser.add_argument("--test-months", type=int, default=12)
    parser.add_argument("--auto-tail-min-target-months", type=int, default=60)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--encoding", default="utf-8-sig")
    return parser.parse_args()


def detail_files(input_dir: Path, start_year: int, end_year: int) -> List[Path]:
    by_year: Dict[int, Path] = {}
    for path in input_dir.glob("StormEvents_details-ftp_v1.0_d*_c*.csv.gz"):
        match = DETAILS_RE.fullmatch(path.name)
        if not match:
            continue
        year = int(match.group("year"))
        if start_year <= year <= end_year:
            by_year[year] = max(path, by_year.get(year, path), key=lambda item: item.name)
    missing = [year for year in range(start_year, end_year + 1) if year not in by_year]
    if missing:
        raise FileNotFoundError(
            f"missing NOAA StormEvents details files for years={missing[:8]}"
            "\nRun benchmarks/noaa_storm_events/download.py first."
        )
    return [by_year[year] for year in range(start_year, end_year + 1)]


def month_from_row(row: Dict[str, str]) -> str:
    value = row.get("BEGIN_YEARMONTH", "").strip()
    if len(value) >= 6 and value[:6].isdigit():
        return f"{value[:4]}-{value[4:6]}"
    date_value = row.get("BEGIN_DATE_TIME", "").strip()
    if len(date_value) >= 7:
        return f"{date_value[:4]}-{date_value[5:7]}"
    raise ValueError(f"cannot infer month from row with BEGIN_YEARMONTH={value!r}")


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    counts: DefaultDict[str, DefaultDict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    state_names: Dict[str, str] = {}
    event_totals = Counter()
    raw_rows_seen = 0

    for path in detail_files(input_dir, args.start_year, args.end_year):
        with gzip.open(path, "rt", encoding=args.encoding, errors="replace", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"STATE", "STATE_FIPS", "EVENT_TYPE", "BEGIN_YEARMONTH"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError(f"{path} is missing required NOAA StormEvents columns")
            for row in reader:
                raw_rows_seen += 1
                state_fips = row["STATE_FIPS"].strip().zfill(2)
                event_type = row["EVENT_TYPE"].strip()
                if not state_fips or not event_type:
                    continue
                month = month_from_row(row)
                counts[state_fips][month][event_type] += 1
                state_names[state_fips] = row["STATE"].strip()
                event_totals[event_type] += 1

    top_k = max(1, args.top_k_event_types)
    top_events = [event for event, _ in event_totals.most_common(top_k)]
    vocab = top_events + ["OTHER"]
    other_index = len(vocab) - 1
    vocab_index = {event: idx for idx, event in enumerate(vocab)}

    sequences: List[torch.Tensor] = []
    train_masks: List[torch.Tensor] = []
    val_masks: List[torch.Tensor] = []
    test_masks: List[torch.Tensor] = []
    entity_rows: List[dict[str, object]] = []
    split_mode_counts = Counter()
    split_target_totals = Counter()

    for state_fips in sorted(counts):
        valid_months = [month for month, event_counts in counts[state_fips].items() if sum(event_counts.values()) > 0]
        if not valid_months:
            continue
        start_month, end_month = longest_contiguous_window(
            valid_months,
            to_index=month_to_index,
            from_index=index_to_month,
        )
        months = month_sequence(start_month, end_month)
        valid_month_set = set(valid_months)
        if len(months) < args.min_contiguous_months or any(month not in valid_month_set for month in months):
            continue

        sequence = torch.zeros((len(months), len(vocab)), dtype=torch.float32)
        for offset, month in enumerate(months):
            for event_type, value in counts[state_fips][month].items():
                idx = vocab_index.get(event_type, other_index)
                sequence[offset, idx] += float(value)
            total = float(sequence[offset].sum().item())
            if total <= 0:
                raise ValueError(f"zero storm event total for state={state_fips} month={month}")
            sequence[offset] /= total

        train_mask, val_mask, test_mask, split_mode = infer_split_masks(
            len(months),
            split_mode=args.split_mode,
            train_fraction=args.train_fraction,
            val_fraction=args.val_fraction,
            val_steps=args.val_months,
            test_steps=args.test_months,
            auto_tail_min_targets=args.auto_tail_min_target_months,
        )
        sequences.append(sequence)
        train_masks.append(train_mask)
        val_masks.append(val_mask)
        test_masks.append(test_mask)
        split_mode_counts[split_mode] += 1
        split_target_totals["train"] += int(train_mask.sum().item())
        split_target_totals["val"] += int(val_mask.sum().item())
        split_target_totals["test"] += int(test_mask.sum().item())
        entity_rows.append(
            {
                "entity_id": state_fips,
                "entity_label": state_names.get(state_fips, state_fips),
                "state": state_names.get(state_fips, ""),
                "state_fips": state_fips,
                "start_month": months[0],
                "end_month": months[-1],
                "length": len(months),
                "train_start": (mask_range(train_mask, months) or {}).get("start", ""),
                "train_end": (mask_range(train_mask, months) or {}).get("end", ""),
                "val_start": (mask_range(val_mask, months) or {}).get("start", ""),
                "val_end": (mask_range(val_mask, months) or {}).get("end", ""),
                "test_start": (mask_range(test_mask, months) or {}).get("start", ""),
                "test_end": (mask_range(test_mask, months) or {}).get("end", ""),
            }
        )

    if not sequences:
        raise ValueError("no NOAA Storm Events sequences survived filtering")

    inputs, lengths = pad_sequences(sequences)
    max_length = inputs.size(1)
    train_loss_mask = torch.zeros((len(sequences), max_length), dtype=torch.bool)
    val_loss_mask = torch.zeros_like(train_loss_mask)
    test_loss_mask = torch.zeros_like(train_loss_mask)
    for idx, length in enumerate(lengths.tolist()):
        train_loss_mask[idx, :length] = train_masks[idx]
        val_loss_mask[idx, :length] = val_masks[idx]
        test_loss_mask[idx, :length] = test_masks[idx]

    save_tensor_splits(
        output_dir,
        inputs=inputs,
        lengths=lengths,
        train_loss_mask=train_loss_mask,
        val_loss_mask=val_loss_mask,
        test_loss_mask=test_loss_mask,
    )
    write_csv(
        output_dir / "vocab.csv",
        [
            {"index": idx, "event_type": event, "token_label": event, "event_count": int(event_totals.get(event, 0))}
            for idx, event in enumerate(vocab)
        ],
        ("index", "event_type", "token_label", "event_count"),
    )
    write_csv(
        output_dir / "entities.csv",
        entity_rows,
        (
            "entity_id",
            "entity_label",
            "state",
            "state_fips",
            "start_month",
            "end_month",
            "length",
            "train_start",
            "train_end",
            "val_start",
            "val_end",
            "test_start",
            "test_end",
        ),
    )
    metadata = {
        "benchmark": TRACK_NAME,
        "description": "NOAA Storm Events state monthly event-type share benchmark.",
        "source": "NOAA Storm Events public yearly details CSV gzip files",
        "source_directory": "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/",
        "access_status": "public direct access; no application or registration required",
        "start_year_requested": args.start_year,
        "end_year_requested": args.end_year,
        "num_sequences": len(sequences),
        "probability_dim": len(vocab),
        "top_k_event_types": args.top_k_event_types,
        "other_token": "OTHER",
        "sequence_length_months_summary": summarize_lengths(lengths.tolist()),
        "split_mode_counts": dict(split_mode_counts),
        "split_target_totals": dict(split_target_totals),
        "raw_rows_seen": raw_rows_seen,
        "artifacts": {
            "vocab_csv": str(output_dir / "vocab.csv"),
            "entities_csv": str(output_dir / "entities.csv"),
        },
    }
    write_json(output_dir / "metadata.json", metadata)
    print(f"[prepare] wrote {output_dir} sequences={len(sequences)} dim={len(vocab)}")


if __name__ == "__main__":
    main()
