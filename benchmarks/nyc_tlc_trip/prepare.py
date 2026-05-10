from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import DefaultDict, Dict, List

import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simplex_benchmark_utils import (
    infer_split_masks,
    mask_range,
    month_sequence,
    pad_sequences,
    save_tensor_splits,
    summarize_lengths,
    write_csv,
    write_json,
)

TRACK_NAME = "nyc_tlc_yellow_pickup_hour_weekly_zone_mix"
DATETIME_COLUMNS = {
    "yellow": "tpep_pickup_datetime",
    "green": "lpep_pickup_datetime",
    "fhv": "pickup_datetime",
    "fhvhv": "pickup_datetime",
}


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parent
    raw_dir = root_dir / "raw"
    parser = argparse.ArgumentParser(description="Prepare an NYC TLC pickup-zone distribution benchmark.")
    parser.add_argument("--input-dir", default=str(raw_dir))
    parser.add_argument("--zone-lookup", default=str(raw_dir / "taxi_zone_lookup.csv"))
    parser.add_argument("--output-dir", default=str(root_dir / "processed"))
    parser.add_argument("--service", choices=("yellow", "green", "fhv", "fhvhv"), default="yellow")
    parser.add_argument("--start-month", default="2023-01")
    parser.add_argument("--end-month", default="2023-12")
    parser.add_argument("--entity", choices=("pickup_hour",), default="pickup_hour")
    parser.add_argument("--min-contiguous-weeks", type=int, default=32)
    parser.add_argument("--min-trips-per-week-entity", type=int, default=1000)
    parser.add_argument("--split-mode", choices=("auto", "tail", "fraction"), default="auto")
    parser.add_argument("--val-weeks", type=int, default=8)
    parser.add_argument("--test-weeks", type=int, default=8)
    parser.add_argument("--auto-tail-min-target-weeks", type=int, default=40)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    return parser.parse_args()


def read_zone_lookup(path: Path) -> Dict[int, Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"missing taxi zone lookup: {path}")
    out: Dict[int, Dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            location_id = int(row["LocationID"])
            out[location_id] = {
                "borough": row.get("Borough", "").strip(),
                "zone": row.get("Zone", "").strip(),
                "service_zone": row.get("service_zone", "").strip(),
            }
    return out


def week_start_label(series):
    return series.dt.to_period("W-SUN").dt.start_time.dt.strftime("%Y-%m-%d")


def week_sequence(start_week: str, end_week: str) -> List[str]:
    start = date.fromisoformat(start_week)
    end = date.fromisoformat(end_week)
    labels = []
    current = start
    while current <= end:
        labels.append(current.isoformat())
        current += timedelta(days=7)
    return labels


def longest_week_window(weeks: List[str]) -> tuple[str, str]:
    ordered = sorted({date.fromisoformat(week) for week in weeks})
    if not ordered:
        raise ValueError("cannot compute a week window for no weeks")
    best_start = best_end = ordered[0]
    cur_start = cur_end = ordered[0]
    for value in ordered[1:]:
        if value == cur_end + timedelta(days=7):
            cur_end = value
            continue
        if (cur_end - cur_start).days > (best_end - best_start).days:
            best_start, best_end = cur_start, cur_end
        cur_start = cur_end = value
    if (cur_end - cur_start).days > (best_end - best_start).days:
        best_start, best_end = cur_start, cur_end
    return best_start.isoformat(), best_end.isoformat()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("NYC TLC preparation requires pandas and pyarrow in the active environment") from exc

    zone_lookup = read_zone_lookup(Path(args.zone_lookup).expanduser().resolve())
    zone_ids = sorted(zone_lookup)
    vocab = [str(location_id) for location_id in zone_ids] + ["UNKNOWN"]
    unknown_index = len(vocab) - 1
    zone_index = {location_id: idx for idx, location_id in enumerate(zone_ids)}

    datetime_col = DATETIME_COLUMNS[args.service]
    counts: DefaultDict[str, DefaultDict[str, Counter[int]]] = defaultdict(lambda: defaultdict(Counter))
    total_rows_seen = 0
    retained_rows = 0

    for month in month_sequence(args.start_month, args.end_month):
        path = input_dir / f"{args.service}_tripdata_{month}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"missing {path}\nRun benchmarks/nyc_tlc_trip/download.py --service {args.service} "
                f"--start-month {args.start_month} --end-month {args.end_month} first."
            )
        frame = pd.read_parquet(path, columns=[datetime_col, "PULocationID"])
        total_rows_seen += int(len(frame))
        frame[datetime_col] = pd.to_datetime(frame[datetime_col], errors="coerce")
        frame = frame.dropna(subset=[datetime_col, "PULocationID"])
        frame = frame[
            (frame[datetime_col].dt.strftime("%Y-%m") >= args.start_month)
            & (frame[datetime_col].dt.strftime("%Y-%m") <= args.end_month)
        ]
        frame["entity_id"] = "hour_" + frame[datetime_col].dt.hour.astype(int).astype(str).str.zfill(2)
        frame["week"] = week_start_label(frame[datetime_col])
        frame["zone_id"] = frame["PULocationID"].fillna(-1).astype(int)
        grouped = frame.groupby(["entity_id", "week", "zone_id"]).size()
        retained_rows += int(grouped.sum())
        for (entity_id, week, zone_id), value in grouped.items():
            counts[str(entity_id)][str(week)][int(zone_id)] += int(value)

    sequences: List[torch.Tensor] = []
    train_masks: List[torch.Tensor] = []
    val_masks: List[torch.Tensor] = []
    test_masks: List[torch.Tensor] = []
    entity_rows: List[dict[str, object]] = []
    split_mode_counts = Counter()
    split_target_totals = Counter()

    for entity_id in sorted(counts):
        valid_weeks = [
            week
            for week, zone_counts in counts[entity_id].items()
            if sum(zone_counts.values()) >= args.min_trips_per_week_entity
        ]
        if not valid_weeks:
            continue
        start_week, end_week = longest_week_window(valid_weeks)
        weeks = week_sequence(start_week, end_week)
        valid_week_set = set(valid_weeks)
        if len(weeks) < args.min_contiguous_weeks or any(week not in valid_week_set for week in weeks):
            continue

        sequence = torch.zeros((len(weeks), len(vocab)), dtype=torch.float32)
        for offset, week in enumerate(weeks):
            for zone_id, value in counts[entity_id][week].items():
                idx = zone_index.get(zone_id, unknown_index)
                sequence[offset, idx] += float(value)
            total = float(sequence[offset].sum().item())
            if total <= 0:
                raise ValueError(f"zero TLC trip total for entity={entity_id} week={week}")
            sequence[offset] /= total

        train_mask, val_mask, test_mask, split_mode = infer_split_masks(
            len(weeks),
            split_mode=args.split_mode,
            train_fraction=args.train_fraction,
            val_fraction=args.val_fraction,
            val_steps=args.val_weeks,
            test_steps=args.test_weeks,
            auto_tail_min_targets=args.auto_tail_min_target_weeks,
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
                "entity_id": entity_id,
                "entity_label": entity_id.replace("_", " "),
                "start_week": weeks[0],
                "end_week": weeks[-1],
                "length": len(weeks),
                "train_start": (mask_range(train_mask, weeks) or {}).get("start", ""),
                "train_end": (mask_range(train_mask, weeks) or {}).get("end", ""),
                "val_start": (mask_range(val_mask, weeks) or {}).get("start", ""),
                "val_end": (mask_range(val_mask, weeks) or {}).get("end", ""),
                "test_start": (mask_range(test_mask, weeks) or {}).get("start", ""),
                "test_end": (mask_range(test_mask, weeks) or {}).get("end", ""),
            }
        )

    if not sequences:
        raise ValueError("no NYC TLC sequences survived filtering")

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
    vocab_rows = []
    for idx, token in enumerate(vocab):
        if token == "UNKNOWN":
            vocab_rows.append({"index": idx, "location_id": token, "token_label": "UNKNOWN", "borough": "", "zone": ""})
            continue
        location_id = int(token)
        info = zone_lookup.get(location_id, {})
        label = f"{location_id} | {info.get('borough', '')} | {info.get('zone', '')}"
        vocab_rows.append(
            {
                "index": idx,
                "location_id": location_id,
                "token_label": label,
                "borough": info.get("borough", ""),
                "zone": info.get("zone", ""),
                "service_zone": info.get("service_zone", ""),
            }
        )
    write_csv(
        output_dir / "vocab.csv",
        vocab_rows,
        ("index", "location_id", "token_label", "borough", "zone", "service_zone"),
    )
    write_csv(
        output_dir / "entities.csv",
        entity_rows,
        (
            "entity_id",
            "entity_label",
            "start_week",
            "end_week",
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
        "description": "NYC TLC weekly pickup-zone share benchmark by pickup hour.",
        "source": "NYC TLC public monthly trip-record Parquet files",
        "source_url_template": "https://d37ci6vzurychx.cloudfront.net/trip-data/{service}_tripdata_{month}.parquet",
        "access_status": "public direct access; no application or registration required",
        "service": args.service,
        "start_month_requested": args.start_month,
        "end_month_requested": args.end_month,
        "entity": args.entity,
        "time_unit": "week",
        "num_sequences": len(sequences),
        "probability_dim": len(vocab),
        "unknown_token": "UNKNOWN",
        "sequence_length_weeks_summary": summarize_lengths(lengths.tolist()),
        "split_mode_counts": dict(split_mode_counts),
        "split_target_totals": dict(split_target_totals),
        "raw_rows_seen": total_rows_seen,
        "retained_grouped_rows": retained_rows,
        "artifacts": {
            "vocab_csv": str(output_dir / "vocab.csv"),
            "entities_csv": str(output_dir / "entities.csv"),
        },
    }
    write_json(output_dir / "metadata.json", metadata)
    print(f"[prepare] wrote {output_dir} sequences={len(sequences)} dim={len(vocab)}")


if __name__ == "__main__":
    main()
