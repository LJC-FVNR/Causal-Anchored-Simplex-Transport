from __future__ import annotations

import argparse
import csv
import zipfile
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

TRACK_NAME = "epa_airdata_county_monthly_aqi_category_mix"
AQI_CATEGORIES = [
    "Good",
    "Moderate",
    "Unhealthy for Sensitive Groups",
    "Unhealthy",
    "Very Unhealthy",
    "Hazardous",
]


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parent
    raw_dir = root_dir / "raw"
    parser = argparse.ArgumentParser(description="Prepare an EPA AirData monthly AQI-category benchmark.")
    parser.add_argument("--input-dir", default=str(raw_dir))
    parser.add_argument("--output-dir", default=str(root_dir / "processed"))
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--min-contiguous-months", type=int, default=60)
    parser.add_argument("--min-valid-days-per-month", type=int, default=20)
    parser.add_argument("--split-mode", choices=("auto", "tail", "fraction"), default="auto")
    parser.add_argument("--val-months", type=int, default=12)
    parser.add_argument("--test-months", type=int, default=12)
    parser.add_argument("--auto-tail-min-target-months", type=int, default=60)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--encoding", default="utf-8-sig")
    return parser.parse_args()


def zip_csv_name(zf: zipfile.ZipFile) -> str:
    names = [name for name in zf.namelist() if name.lower().endswith(".csv")]
    if not names:
        raise ValueError("EPA AirData zip did not contain a CSV file")
    return names[0]


def month_from_date(date_text: str) -> str:
    return date_text.strip()[:7]


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    counts: DefaultDict[str, DefaultDict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    county_names: Dict[str, str] = {}
    state_names: Dict[str, str] = {}
    raw_rows_seen = 0
    retained_rows = 0

    for year in range(args.start_year, args.end_year + 1):
        path = input_dir / f"daily_aqi_by_county_{year}.zip"
        if not path.exists():
            raise FileNotFoundError(
                f"missing {path}\nRun benchmarks/epa_airdata_aqi/download.py --start-year {args.start_year} --end-year {args.end_year} first."
            )
        with zipfile.ZipFile(path) as zf:
            with zf.open(zip_csv_name(zf), "r") as raw_handle:
                text_handle = (line.decode(args.encoding, errors="replace") for line in raw_handle)
                reader = csv.DictReader(text_handle)
                required = {"State Name", "county Name", "State Code", "County Code", "Date", "Category"}
                if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                    raise ValueError(f"{path} is missing required EPA AirData columns")
                for row in reader:
                    raw_rows_seen += 1
                    category = row["Category"].strip()
                    if category not in AQI_CATEGORIES:
                        continue
                    state_code = row["State Code"].strip().zfill(2)
                    county_code = row["County Code"].strip().zfill(3)
                    county_fips = f"{state_code}{county_code}"
                    month = month_from_date(row["Date"])
                    counts[county_fips][month][category] += 1
                    county_names[county_fips] = row["county Name"].strip()
                    state_names[county_fips] = row["State Name"].strip()
                    retained_rows += 1

    vocab_index = {category: idx for idx, category in enumerate(AQI_CATEGORIES)}
    sequences: List[torch.Tensor] = []
    train_masks: List[torch.Tensor] = []
    val_masks: List[torch.Tensor] = []
    test_masks: List[torch.Tensor] = []
    entity_rows: List[dict[str, object]] = []
    split_mode_counts = Counter()
    split_target_totals = Counter()

    for county_fips in sorted(counts):
        valid_months = [
            month
            for month, category_counts in counts[county_fips].items()
            if sum(category_counts.values()) >= args.min_valid_days_per_month
        ]
        if not valid_months:
            continue
        start_month, end_month = longest_contiguous_window(
            valid_months,
            to_index=month_to_index,
            from_index=index_to_month,
        )
        months = month_sequence(start_month, end_month)
        if len(months) < args.min_contiguous_months:
            continue
        if any(month not in valid_months for month in months):
            continue

        sequence = torch.zeros((len(months), len(AQI_CATEGORIES)), dtype=torch.float32)
        for offset, month in enumerate(months):
            for category, value in counts[county_fips][month].items():
                sequence[offset, vocab_index[category]] += float(value)
            total = float(sequence[offset].sum().item())
            if total <= 0:
                raise ValueError(f"zero AQI total for county={county_fips} month={month}")
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
                "entity_id": county_fips,
                "entity_label": f"{state_names.get(county_fips, '')} | {county_names.get(county_fips, county_fips)}",
                "state": state_names.get(county_fips, ""),
                "county": county_names.get(county_fips, ""),
                "county_fips": county_fips,
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
        raise ValueError("no EPA AQI county sequences survived filtering")

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
        [{"index": idx, "token_label": label, "ordered_support": idx} for idx, label in enumerate(AQI_CATEGORIES)],
        ("index", "token_label", "ordered_support"),
    )
    write_csv(
        output_dir / "entities.csv",
        entity_rows,
        (
            "entity_id",
            "entity_label",
            "state",
            "county",
            "county_fips",
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
        "description": "EPA AirData county monthly AQI-category share benchmark.",
        "source": "EPA AirData daily_aqi_by_county public CSV zips",
        "source_url_template": "https://aqs.epa.gov/aqsweb/airdata/daily_aqi_by_county_{year}.zip",
        "access_status": "public direct access; no application or registration required for static CSV/ZIP files",
        "start_year_requested": args.start_year,
        "end_year_requested": args.end_year,
        "num_sequences": len(sequences),
        "probability_dim": len(AQI_CATEGORIES),
        "ordered_support": True,
        "aqi_categories": AQI_CATEGORIES,
        "min_valid_days_per_month": args.min_valid_days_per_month,
        "sequence_length_months_summary": summarize_lengths(lengths.tolist()),
        "split_mode_counts": dict(split_mode_counts),
        "split_target_totals": dict(split_target_totals),
        "raw_rows_seen": raw_rows_seen,
        "retained_rows": retained_rows,
        "artifacts": {
            "vocab_csv": str(output_dir / "vocab.csv"),
            "entities_csv": str(output_dir / "entities.csv"),
        },
    }
    write_json(output_dir / "metadata.json", metadata)
    print(f"[prepare] wrote {output_dir} sequences={len(sequences)} dim={len(AQI_CATEGORIES)}")


if __name__ == "__main__":
    main()
