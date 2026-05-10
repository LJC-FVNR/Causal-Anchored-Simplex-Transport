from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from owid_dietary_common import FOOD_COLUMNS, TRACK_NAME, TRAIN_FRACTION, VAL_FRACTION, WINDOW_END_YEAR, WINDOW_START_YEAR


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parent
    raw_dir = root_dir / "raw"
    parser = argparse.ArgumentParser(
        description="Prepare an OWID dietary-composition benchmark in tensor_dir format."
    )
    parser.add_argument(
        "--input",
        default=str(raw_dir / "dietary-composition-by-country.csv"),
        help="Path to the OWID grapher CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(root_dir / "processed"),
        help="Directory where train.pt / val.pt / test.pt and metadata.json will be written.",
    )
    parser.add_argument("--start-year", type=int, default=WINDOW_START_YEAR)
    parser.add_argument("--end-year", type=int, default=WINDOW_END_YEAR)
    parser.add_argument("--train-fraction", type=float, default=TRAIN_FRACTION)
    parser.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    parser.add_argument(
        "--encoding",
        default="utf-8-sig",
        help="Text encoding for the OWID CSV input. Defaults to utf-8-sig.",
    )
    return parser.parse_args()


def parse_optional_float(value: str) -> Optional[float]:
    text = str(value).strip().replace(",", "")
    if text in {"", "NA", "NaN", "nan", "."}:
        return None
    return float(text)


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

    train_mask = torch.zeros(length, dtype=torch.bool)
    val_mask = torch.zeros(length, dtype=torch.bool)
    test_mask = torch.zeros(length, dtype=torch.bool)
    train_end = 1 + train_count
    val_end = train_end + val_count
    train_mask[1:train_end] = True
    val_mask[train_end:val_end] = True
    test_mask[val_end:length] = True
    return train_mask, val_mask, test_mask, (train_count, val_count, test_count)


def year_range_from_mask(mask: torch.Tensor, years: List[int]) -> Optional[Dict[str, int]]:
    indices = torch.nonzero(mask, as_tuple=False).squeeze(-1)
    if indices.numel() == 0:
        return None
    start_idx = int(indices[0].item())
    end_idx = int(indices[-1].item())
    return {
        "start_year": years[start_idx],
        "end_year": years[end_idx],
        "num_target_steps": int(indices.numel()),
    }


def is_current_iso3_entity(code: str) -> bool:
    return len(code) == 3 and not code.startswith("OWID_")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not (args.start_year < args.end_year):
        raise ValueError("expected start-year < end-year")
    if not input_path.exists():
        raise FileNotFoundError(
            f"missing input file: {input_path}\n"
            "Run benchmarks/owid_dietary/download.py first, or point --input to the OWID grapher CSV."
        )

    years = list(range(args.start_year, args.end_year + 1))
    entity_rows: Dict[str, Dict[int, List[float]]] = {}
    entity_names: Dict[str, str] = {}
    blank_counts: Dict[str, Dict[int, int]] = {}
    total_kcal_raw: Dict[str, Dict[int, float]] = {}

    with input_path.open("r", encoding=args.encoding, newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{input_path} is missing a CSV header row")

        required_columns = {"Entity", "Code", "Year", *FOOD_COLUMNS}
        missing = required_columns.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"{input_path} is missing OWID columns: {sorted(missing)}")

        for row in reader:
            code = row["Code"].strip()
            if not is_current_iso3_entity(code):
                continue
            year_raw = parse_optional_float(row["Year"])
            if year_raw is None:
                continue
            year = int(year_raw)
            if year < args.start_year or year > args.end_year:
                continue

            entity = row["Entity"].strip()
            values: List[float] = []
            blank_count = 0
            for column in FOOD_COLUMNS:
                parsed = parse_optional_float(row[column])
                if parsed is None:
                    values.append(0.0)
                    blank_count += 1
                else:
                    values.append(parsed)
            total = float(sum(values))
            if total <= 0:
                continue

            entity_names[code] = entity
            entity_rows.setdefault(code, {})[year] = values
            blank_counts.setdefault(code, {})[year] = blank_count
            total_kcal_raw.setdefault(code, {})[year] = total

    selected_codes = sorted(code for code, row_map in entity_rows.items() if sorted(row_map) == years)
    if not selected_codes:
        raise ValueError(
            f"No OWID entities have a complete current-ISO3 annual panel over {args.start_year}..{args.end_year}."
        )

    length = len(years)
    train_mask_1d, val_mask_1d, test_mask_1d, split_counts = build_fraction_masks(
        length,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
    )
    train_range = year_range_from_mask(train_mask_1d, years)
    val_range = year_range_from_mask(val_mask_1d, years)
    test_range = year_range_from_mask(test_mask_1d, years)

    num_entities = len(selected_codes)
    dim = len(FOOD_COLUMNS)
    inputs = torch.zeros((num_entities, length, dim), dtype=torch.float32)
    train_loss_mask = train_mask_1d.unsqueeze(0).expand(num_entities, -1).clone()
    val_loss_mask = val_mask_1d.unsqueeze(0).expand(num_entities, -1).clone()
    test_loss_mask = test_mask_1d.unsqueeze(0).expand(num_entities, -1).clone()
    total_kcal_capita_day = torch.zeros((num_entities, length), dtype=torch.float32)
    blank_component_count = torch.zeros((num_entities, length), dtype=torch.int16)
    observation_year = torch.tensor([years for _ in range(num_entities)], dtype=torch.int32)
    lengths = torch.full((num_entities,), length, dtype=torch.long)

    rows = []
    totals_all = []
    blanks_all = []

    for idx, code in enumerate(selected_codes):
        entity = entity_names[code]
        for year_idx, year in enumerate(years):
            values = entity_rows[code][year]
            total = total_kcal_raw[code][year]
            vector = torch.tensor(values, dtype=torch.float32)
            inputs[idx, year_idx] = vector / float(vector.sum().item())
            total_kcal_capita_day[idx, year_idx] = total
            blank_component_count[idx, year_idx] = blank_counts[code][year]
            totals_all.append(total)
            blanks_all.append(blank_counts[code][year])

        rows.append(
            {
                "sequence_index": idx,
                "entity": entity,
                "code": code,
                "start_year": years[0],
                "end_year": years[-1],
                "sequence_length_years": length,
                "train_target_start_year": train_range["start_year"] if train_range else "",
                "train_target_end_year": train_range["end_year"] if train_range else "",
                "train_target_steps": train_range["num_target_steps"] if train_range else 0,
                "val_target_start_year": val_range["start_year"] if val_range else "",
                "val_target_end_year": val_range["end_year"] if val_range else "",
                "val_target_steps": val_range["num_target_steps"] if val_range else 0,
                "test_target_start_year": test_range["start_year"] if test_range else "",
                "test_target_end_year": test_range["end_year"] if test_range else "",
                "test_target_steps": test_range["num_target_steps"] if test_range else 0,
            }
        )

    payload_common = {
        "inputs": inputs,
        "lengths": lengths,
        "total_kcal_capita_day": total_kcal_capita_day,
        "blank_component_count": blank_component_count,
        "observation_year": observation_year,
    }
    torch.save({**payload_common, "loss_mask": train_loss_mask}, output_dir / "train.pt")
    torch.save({**payload_common, "loss_mask": val_loss_mask}, output_dir / "val.pt")
    torch.save({**payload_common, "loss_mask": test_loss_mask}, output_dir / "test.pt")

    vocab_csv_path = output_dir / "vocab.csv"
    with vocab_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["index", "group_label"])
        writer.writeheader()
        for idx, label in enumerate(FOOD_COLUMNS):
            writer.writerow({"index": idx, "group_label": label})

    entities_csv_path = output_dir / "entities.csv"
    with entities_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "benchmark": TRACK_NAME,
        "description": (
            "Annual country dietary-composition benchmark derived from the OWID dietary-composition grapher, "
            "restricted to current ISO3 entities with a complete 1961-2023 panel."
        ),
        "source_input": str(input_path),
        "num_entities": num_entities,
        "num_years": length,
        "probability_dim": dim,
        "food_columns": FOOD_COLUMNS,
        "time_unit": "year",
        "sequence_unit": "country",
        "window_start_year": args.start_year,
        "window_end_year": args.end_year,
        "selected_entity_codes": selected_codes,
        "train_fraction": args.train_fraction,
        "val_fraction": args.val_fraction,
        "train_target_years": list(range(train_range["start_year"], train_range["end_year"] + 1)) if train_range else [],
        "val_target_years": list(range(val_range["start_year"], val_range["end_year"] + 1)) if val_range else [],
        "test_target_years": list(range(test_range["start_year"], test_range["end_year"] + 1)) if test_range else [],
        "split_counts": {
            "train_targets": split_counts[0],
            "val_targets": split_counts[1],
            "test_targets": split_counts[2],
        },
        "sequence_length_years_summary": summarize_lengths([length] * num_entities),
        "total_kcal_capita_day_summary": {
            "min": min(totals_all),
            "median": sorted(totals_all)[len(totals_all) // 2],
            "mean": sum(totals_all) / len(totals_all),
            "max": max(totals_all),
        },
        "blank_component_count_summary": {
            "min": min(blanks_all),
            "median": sorted(blanks_all)[len(blanks_all) // 2],
            "mean": sum(blanks_all) / len(blanks_all),
            "max": max(blanks_all),
        },
        "split_rationale": (
            "Keep only current ISO3 entities with a complete 1961-2023 panel to avoid mixed historical fragments "
            "and apply a fixed chronological 70/15/15 split on one-step target years."
        ),
        "artifacts": {
            "vocab_csv": str(vocab_csv_path),
            "entities_csv": str(entities_csv_path),
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"[prepare] wrote {output_dir}")
    print(
        f"[prepare] entities={num_entities} years={length} dim={dim} "
        f"train={train_range['start_year']}..{train_range['end_year']} "
        f"val={val_range['start_year']}..{val_range['end_year']} "
        f"test={test_range['start_year']}..{test_range['end_year']}"
    )


if __name__ == "__main__":
    main()

