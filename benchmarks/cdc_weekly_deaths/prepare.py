from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional, Tuple

import torch

from cdc_weekly_common import (
    AGGREGATED_NEW_YORK_NAME,
    EXCLUDED_JURISDICTIONS,
    NEW_YORK_COMPONENTS,
    OTHER_TOKEN,
    SELECTED_CAUSE_COLUMNS,
    SUPPRESSED_MIDPOINT,
    TAIL_DROP_WEEKS,
    TEST_WEEKS,
    TRACK_NAME,
    TRAIN_FRACTION,
    VAL_FRACTION,
    VAL_WEEKS,
)

WeekKey = Tuple[int, int, str]


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parent
    raw_dir = root_dir / "raw"
    parser = argparse.ArgumentParser(
        description="Prepare a CDC weekly cause-share benchmark in tensor_dir format."
    )
    parser.add_argument(
        "--input",
        default=str(raw_dir / "cdc_weekly_deaths.csv"),
        help="Path to the CDC weekly provisional deaths CSV.",
    )
    parser.add_argument(
        "--metadata-json",
        default=str(raw_dir / "cdc_weekly_deaths.metadata.json"),
        help="Optional CDC Socrata metadata JSON path for benchmark provenance.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(root_dir / "processed"),
        help="Directory where train.pt / val.pt / test.pt and metadata.json will be written.",
    )
    parser.add_argument(
        "--split-mode",
        default="auto",
        choices=("auto", "tail", "fraction"),
        help="Use tail holdout when enough history exists, otherwise fall back to fraction split.",
    )
    parser.add_argument("--tail-drop-weeks", type=int, default=TAIL_DROP_WEEKS)
    parser.add_argument("--val-weeks", type=int, default=VAL_WEEKS)
    parser.add_argument("--test-weeks", type=int, default=TEST_WEEKS)
    parser.add_argument("--train-fraction", type=float, default=TRAIN_FRACTION)
    parser.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    parser.add_argument(
        "--suppressed-midpoint",
        type=float,
        default=SUPPRESSED_MIDPOINT,
        help="Imputed count used when CDC suppresses a retained cause as 1-9.",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8-sig",
        help="Text encoding for CSV reads. Defaults to utf-8-sig.",
    )
    return parser.parse_args()


def parse_optional_float(value: str) -> Optional[float]:
    text = str(value).strip().replace(",", "")
    if text in {"", "NA", "NaN", "nan", "."}:
        return None
    return float(text)


def normalize_jurisdiction(name: str) -> Optional[str]:
    stripped = name.strip()
    if stripped in EXCLUDED_JURISDICTIONS:
        return None
    if stripped in NEW_YORK_COMPONENTS:
        return AGGREGATED_NEW_YORK_NAME
    return stripped


def summarize_lengths(lengths: List[int]) -> Dict[str, float]:
    ordered = sorted(lengths)
    n = len(ordered)
    return {
        "min": float(ordered[0]),
        "median": float(ordered[n // 2]),
        "mean": float(sum(ordered) / n),
        "max": float(ordered[-1]),
    }


def build_fraction_masks(length: int, *, train_fraction: float, val_fraction: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[int, int, int]]:
    if length < 4:
        raise ValueError(f"sequence length must be at least 4 to form a meaningful split, got {length}")
    if train_fraction <= 0 or val_fraction <= 0 or train_fraction + val_fraction >= 1:
        raise ValueError("expected train_fraction > 0, val_fraction > 0, and train_fraction + val_fraction < 1")

    num_targets = length - 1
    train_count = max(1, int(num_targets * train_fraction))
    val_count = max(1, int(num_targets * val_fraction))
    test_count = num_targets - train_count - val_count

    if test_count < 1:
        deficit = 1 - test_count
        train_reduction = min(deficit, max(0, train_count - 1))
        train_count -= train_reduction
        deficit -= train_reduction
        if deficit > 0:
            val_reduction = min(deficit, max(0, val_count - 1))
            val_count -= val_reduction
            deficit -= val_reduction
        test_count = num_targets - train_count - val_count

    if min(train_count, val_count, test_count) < 1:
        raise ValueError(
            f"could not form a valid split from {num_targets} targets: "
            f"train={train_count}, val={val_count}, test={test_count}"
        )

    return masks_from_counts(length, train_count=train_count, val_count=val_count, test_count=test_count)


def masks_from_counts(length: int, *, train_count: int, val_count: int, test_count: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[int, int, int]]:
    if train_count + val_count + test_count != length - 1:
        raise ValueError("expected train_count + val_count + test_count == length - 1")

    train_mask = torch.zeros(length, dtype=torch.bool)
    val_mask = torch.zeros(length, dtype=torch.bool)
    test_mask = torch.zeros(length, dtype=torch.bool)
    train_end = 1 + train_count
    val_end = train_end + val_count
    train_mask[1:train_end] = True
    val_mask[train_end:val_end] = True
    test_mask[val_end:length] = True
    return train_mask, val_mask, test_mask, (train_count, val_count, test_count)


def build_tail_masks(length: int, *, val_weeks: int, test_weeks: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[int, int, int]]:
    num_targets = length - 1
    if val_weeks < 1 or test_weeks < 1:
        raise ValueError("expected val_weeks >= 1 and test_weeks >= 1")
    train_count = num_targets - val_weeks - test_weeks
    if train_count < 1:
        raise ValueError(
            f"not enough history for tail split: length={length}, num_targets={num_targets}, "
            f"val_weeks={val_weeks}, test_weeks={test_weeks}"
        )
    return masks_from_counts(length, train_count=train_count, val_count=val_weeks, test_count=test_weeks)


def week_range_from_mask(mask: torch.Tensor, weeks: List[WeekKey]) -> Optional[Dict[str, object]]:
    indices = torch.nonzero(mask, as_tuple=False).squeeze(-1)
    if indices.numel() == 0:
        return None
    start_idx = int(indices[0].item())
    end_idx = int(indices[-1].item())
    start_year, start_week, start_date = weeks[start_idx]
    end_year, end_week, end_date = weeks[end_idx]
    return {
        "start_mmwr_year": start_year,
        "start_mmwr_week": start_week,
        "start_week_ending_date": start_date,
        "end_mmwr_year": end_year,
        "end_mmwr_week": end_week,
        "end_week_ending_date": end_date,
        "num_target_steps": int(indices.numel()),
    }


def select_split(
    length: int,
    *,
    split_mode: str,
    val_weeks: int,
    test_weeks: int,
    train_fraction: float,
    val_fraction: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[int, int, int], str]:
    if split_mode == "tail":
        train_mask, val_mask, test_mask, counts = build_tail_masks(length, val_weeks=val_weeks, test_weeks=test_weeks)
        return train_mask, val_mask, test_mask, counts, "tail"
    if split_mode == "fraction":
        train_mask, val_mask, test_mask, counts = build_fraction_masks(length, train_fraction=train_fraction, val_fraction=val_fraction)
        return train_mask, val_mask, test_mask, counts, "fraction"

    num_targets = length - 1
    if num_targets > val_weeks + test_weeks:
        train_mask, val_mask, test_mask, counts = build_tail_masks(length, val_weeks=val_weeks, test_weeks=test_weeks)
        return train_mask, val_mask, test_mask, counts, "tail"
    train_mask, val_mask, test_mask, counts = build_fraction_masks(length, train_fraction=train_fraction, val_fraction=val_fraction)
    return train_mask, val_mask, test_mask, counts, "fraction"


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    metadata_json_path = Path(args.metadata_json).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.tail_drop_weeks < 0:
        raise ValueError("--tail-drop-weeks must be >= 0")
    if not input_path.exists():
        raise FileNotFoundError(
            f"missing input file: {input_path}\n"
            "Run benchmarks/cdc_weekly_deaths/download.py first, or point --input to the CDC CSV."
        )

    raw_rows = []
    with input_path.open("r", encoding=args.encoding, newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{input_path} is missing a CSV header row")
        required = {
            "Data As Of",
            "Jurisdiction of Occurrence",
            "MMWR Year",
            "MMWR Week",
            "Week Ending Date",
            "All Cause",
            *(source_col for _label, source_col, _flag in SELECTED_CAUSE_COLUMNS),
            *(flag_col for _label, _source_col, flag_col in SELECTED_CAUSE_COLUMNS),
        }
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"{input_path} is missing CDC columns: {sorted(missing)}")
        raw_rows = list(reader)

    snapshot_dates = sorted({row["Data As Of"].strip() for row in raw_rows if row["Data As Of"].strip()})
    weeks_all = sorted(
        {
            (int(row["MMWR Year"]), int(row["MMWR Week"]), row["Week Ending Date"].strip())
            for row in raw_rows
            if normalize_jurisdiction(row["Jurisdiction of Occurrence"]) is not None
        }
    )
    if args.tail_drop_weeks >= len(weeks_all):
        raise ValueError(
            f"cannot drop {args.tail_drop_weeks} tail weeks from a CDC history of only {len(weeks_all)} weeks"
        )
    kept_weeks = weeks_all[: len(weeks_all) - args.tail_drop_weeks] if args.tail_drop_weeks > 0 else list(weeks_all)
    kept_week_keys = {(year, week) for year, week, _date in kept_weeks}

    aggregate: DefaultDict[Tuple[str, int, int], Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    states_set = set()
    for row in raw_rows:
        state = normalize_jurisdiction(row["Jurisdiction of Occurrence"])
        if state is None:
            continue
        mmwr_year = int(row["MMWR Year"])
        mmwr_week = int(row["MMWR Week"])
        if (mmwr_year, mmwr_week) not in kept_week_keys:
            continue

        all_cause = parse_optional_float(row["All Cause"])
        if all_cause is None:
            continue
        states_set.add(state)
        bucket = aggregate[(state, mmwr_year, mmwr_week)]
        bucket["all_cause"] += all_cause

        for cause_label, source_col, flag_col in SELECTED_CAUSE_COLUMNS:
            raw_value = parse_optional_float(row[source_col])
            if raw_value is not None:
                bucket[cause_label] += raw_value
            elif row[flag_col].strip():
                bucket[cause_label] += args.suppressed_midpoint
                bucket["suppressed_selected_count"] += 1.0

    states = sorted(states_set)
    if len(states) != 50:
        raise ValueError(f"expected 50 retained states after CDC aggregation, found {len(states)}: {states}")

    for state in states:
        available = {(year, week) for (row_state, year, week) in aggregate if row_state == state}
        if available != kept_week_keys:
            missing = sorted(kept_week_keys.difference(available))
            raise ValueError(f"state={state} is missing retained CDC weeks, e.g. {missing[:5]}")

    length = len(kept_weeks)
    train_mask_1d, val_mask_1d, test_mask_1d, split_counts, resolved_split_mode = select_split(
        length,
        split_mode=args.split_mode,
        val_weeks=args.val_weeks,
        test_weeks=args.test_weeks,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
    )
    train_range = week_range_from_mask(train_mask_1d, kept_weeks)
    val_range = week_range_from_mask(val_mask_1d, kept_weeks)
    test_range = week_range_from_mask(test_mask_1d, kept_weeks)

    dim = len(SELECTED_CAUSE_COLUMNS) + 1
    num_states = len(states)
    inputs = torch.zeros((num_states, length, dim), dtype=torch.float32)
    train_loss_mask = train_mask_1d.unsqueeze(0).expand(num_states, -1).clone()
    val_loss_mask = val_mask_1d.unsqueeze(0).expand(num_states, -1).clone()
    test_loss_mask = test_mask_1d.unsqueeze(0).expand(num_states, -1).clone()
    total_weekly_deaths = torch.zeros((num_states, length), dtype=torch.float32)
    other_causes_count = torch.zeros((num_states, length), dtype=torch.float32)
    suppressed_selected_count = torch.zeros((num_states, length), dtype=torch.int16)
    mmwr_year = torch.tensor([[week[0] for week in kept_weeks] for _ in range(num_states)], dtype=torch.int16)
    mmwr_week = torch.tensor([[week[1] for week in kept_weeks] for _ in range(num_states)], dtype=torch.int16)
    lengths = torch.full((num_states,), length, dtype=torch.long)

    rows = []
    selected_sum_total = 0.0
    all_cause_total = 0.0
    residual_clip_count = 0
    suppressed_cells_total = 0.0

    for state_idx, state in enumerate(states):
        for week_idx, (year, week, _week_ending_date) in enumerate(kept_weeks):
            bucket = aggregate[(state, year, week)]
            total = float(bucket["all_cause"])
            selected_values = [float(bucket[cause_label]) for cause_label, _source_col, _flag_col in SELECTED_CAUSE_COLUMNS]
            selected_sum = sum(selected_values)
            residual = total - selected_sum
            if residual < 0:
                residual = 0.0
                residual_clip_count += 1

            vector = torch.tensor(selected_values + [residual], dtype=torch.float32)
            if float(vector.sum().item()) <= 0:
                raise ValueError(f"non-positive CDC cause vector for state={state} year={year} week={week}")

            inputs[state_idx, week_idx] = vector / float(vector.sum().item())
            total_weekly_deaths[state_idx, week_idx] = total
            other_causes_count[state_idx, week_idx] = residual
            suppressed_selected_count[state_idx, week_idx] = int(bucket.get("suppressed_selected_count", 0.0))

            selected_sum_total += selected_sum
            all_cause_total += total
            suppressed_cells_total += bucket.get("suppressed_selected_count", 0.0)

        rows.append(
            {
                "sequence_index": state_idx,
                "state": state,
                "start_mmwr_year": kept_weeks[0][0],
                "start_mmwr_week": kept_weeks[0][1],
                "end_mmwr_year": kept_weeks[-1][0],
                "end_mmwr_week": kept_weeks[-1][1],
                "sequence_length_weeks": length,
                "train_targets": split_counts[0],
                "val_targets": split_counts[1],
                "test_targets": split_counts[2],
                "train_target_start_mmwr_year": train_range["start_mmwr_year"] if train_range else "",
                "train_target_start_mmwr_week": train_range["start_mmwr_week"] if train_range else "",
                "train_target_end_mmwr_year": train_range["end_mmwr_year"] if train_range else "",
                "train_target_end_mmwr_week": train_range["end_mmwr_week"] if train_range else "",
                "val_target_start_mmwr_year": val_range["start_mmwr_year"] if val_range else "",
                "val_target_start_mmwr_week": val_range["start_mmwr_week"] if val_range else "",
                "val_target_end_mmwr_year": val_range["end_mmwr_year"] if val_range else "",
                "val_target_end_mmwr_week": val_range["end_mmwr_week"] if val_range else "",
                "test_target_start_mmwr_year": test_range["start_mmwr_year"] if test_range else "",
                "test_target_start_mmwr_week": test_range["start_mmwr_week"] if test_range else "",
                "test_target_end_mmwr_year": test_range["end_mmwr_year"] if test_range else "",
                "test_target_end_mmwr_week": test_range["end_mmwr_week"] if test_range else "",
            }
        )

    payload_common = {
        "inputs": inputs,
        "lengths": lengths,
        "total_weekly_deaths": total_weekly_deaths,
        "other_causes_count": other_causes_count,
        "suppressed_selected_count": suppressed_selected_count,
        "mmwr_year": mmwr_year,
        "mmwr_week": mmwr_week,
    }
    torch.save({**payload_common, "loss_mask": train_loss_mask}, output_dir / "train.pt")
    torch.save({**payload_common, "loss_mask": val_loss_mask}, output_dir / "val.pt")
    torch.save({**payload_common, "loss_mask": test_loss_mask}, output_dir / "test.pt")

    vocab_csv_path = output_dir / "vocab.csv"
    with vocab_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["index", "token_label", "source_column"])
        writer.writeheader()
        for idx, (cause_label, source_col, _flag_col) in enumerate(SELECTED_CAUSE_COLUMNS):
            writer.writerow({"index": idx, "token_label": cause_label, "source_column": source_col})
        writer.writerow({"index": len(SELECTED_CAUSE_COLUMNS), "token_label": OTHER_TOKEN, "source_column": "All Cause residual"})

    states_csv_path = output_dir / "states.csv"
    with states_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    weeks_csv_path = output_dir / "weeks.csv"
    with weeks_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "index",
                "mmwr_year",
                "mmwr_week",
                "week_ending_date",
                "train_target",
                "val_target",
                "test_target",
            ],
        )
        writer.writeheader()
        for idx, (year, week, week_ending_date) in enumerate(kept_weeks):
            writer.writerow(
                {
                    "index": idx,
                    "mmwr_year": year,
                    "mmwr_week": week,
                    "week_ending_date": week_ending_date,
                    "train_target": int(train_mask_1d[idx].item()),
                    "val_target": int(val_mask_1d[idx].item()),
                    "test_target": int(test_mask_1d[idx].item()),
                }
            )

    metadata_payload = {}
    if metadata_json_path.exists():
        try:
            metadata_payload = json.loads(metadata_json_path.read_text(encoding="utf-8"))
        except Exception:
            metadata_payload = {}

    metadata = {
        "benchmark": TRACK_NAME,
        "description": (
            "Weekly state-level cause-share benchmark derived from CDC provisional deaths data, using five low-suppression "
            "selected underlying causes plus an OTHER residual over all-cause deaths."
        ),
        "source_input": str(input_path),
        "source_snapshot_dates": snapshot_dates,
        "socrata_name": metadata_payload.get("name"),
        "socrata_description": metadata_payload.get("description"),
        "num_states": num_states,
        "num_weeks": length,
        "probability_dim": dim,
        "selected_cause_labels": [label for label, _source_col, _flag_col in SELECTED_CAUSE_COLUMNS],
        "other_token": OTHER_TOKEN,
        "excluded_jurisdictions": sorted(EXCLUDED_JURISDICTIONS),
        "new_york_aggregation": "New York and New York City were summed into a single New York state sequence.",
        "tail_drop_weeks": args.tail_drop_weeks,
        "resolved_split_mode": resolved_split_mode,
        "val_weeks": args.val_weeks,
        "test_weeks": args.test_weeks,
        "train_fraction": args.train_fraction,
        "val_fraction": args.val_fraction,
        "sequence_length_weeks_summary": summarize_lengths([length] * num_states),
        "selected_cause_share_of_all_cause": selected_sum_total / all_cause_total if all_cause_total > 0 else 0.0,
        "suppressed_midpoint": args.suppressed_midpoint,
        "suppressed_selected_cells_total": suppressed_cells_total,
        "suppressed_selected_cells_per_state_week_mean": suppressed_cells_total / (num_states * length),
        "negative_residual_clip_count": residual_clip_count,
        "train_target_range": train_range,
        "val_target_range": val_range,
        "test_target_range": test_range,
        "split_rationale": (
            "Use the frozen 2020-2023 CDC weekly deaths snapshot, aggregate New York and New York City into a full-state "
            "New York series, exclude the final tail weeks of the frozen provisional snapshot to reduce end-of-series "
            "under-reporting, then apply a tail holdout of 52 validation and 52 test target weeks when history is long enough."
        ),
        "artifacts": {
            "vocab_csv": str(vocab_csv_path),
            "states_csv": str(states_csv_path),
            "weeks_csv": str(weeks_csv_path),
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"[prepare] wrote {output_dir}")
    print(
        f"[prepare] states={num_states} weeks={length} dim={dim} "
        f"selected_share={metadata['selected_cause_share_of_all_cause']:.4f}"
    )
    if train_range and val_range and test_range:
        print(
            f"[prepare] split train={train_range['start_mmwr_year']}W{train_range['start_mmwr_week']:02d}.."
            f"{train_range['end_mmwr_year']}W{train_range['end_mmwr_week']:02d} "
            f"val={val_range['start_mmwr_year']}W{val_range['start_mmwr_week']:02d}.."
            f"{val_range['end_mmwr_year']}W{val_range['end_mmwr_week']:02d} "
            f"test={test_range['start_mmwr_year']}W{test_range['start_mmwr_week']:02d}.."
            f"{test_range['end_mmwr_year']}W{test_range['end_mmwr_week']:02d}"
        )


if __name__ == "__main__":
    main()

