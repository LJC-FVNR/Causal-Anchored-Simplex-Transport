from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional, Tuple

import torch

TRACK_NAME = "ember_monthly_generation_mix"
FUEL_VOCAB = [
    "Coal",
    "Gas",
    "Other Fossil",
    "Nuclear",
    "Hydro",
    "Wind",
    "Solar",
    "Bioenergy",
    "Other Renewables",
]
CountryKey = str


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parent
    raw_dir = root_dir / "raw"
    parser = argparse.ArgumentParser(
        description="Prepare an Ember monthly electricity generation-mix benchmark in tensor_dir format."
    )
    parser.add_argument(
        "--input",
        default=str(raw_dir / "monthly_full_release_long_format.csv"),
        help="Path to Ember's official monthly long-format CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(root_dir / "processed"),
        help="Directory where train.pt / val.pt / test.pt and metadata.json will be written.",
    )
    parser.add_argument(
        "--split-mode",
        choices=("auto", "tail", "fraction"),
        default="auto",
        help="auto -> use final 24 target months (12 val + 12 test) when the sequence is long enough, otherwise 70/15/15.",
    )
    parser.add_argument(
        "--val-months",
        type=int,
        default=12,
        help="Number of target months reserved for validation when split-mode is auto or tail.",
    )
    parser.add_argument(
        "--test-months",
        type=int,
        default=12,
        help="Number of target months reserved for test when split-mode is auto or tail.",
    )
    parser.add_argument(
        "--auto-tail-min-target-months",
        type=int,
        default=60,
        help="Minimum number of one-step target months required before auto mode uses the fixed tail split.",
    )
    parser.add_argument(
        "--min-contiguous-months",
        type=int,
        default=36,
        help="Minimum length of the longest contiguous monthly country window kept in the benchmark.",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.70,
        help="Chronological train fraction applied when split-mode resolves to fraction.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.15,
        help="Chronological validation fraction applied when split-mode resolves to fraction.",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8-sig",
        help="Text encoding for the Ember CSV input. Defaults to utf-8-sig.",
    )
    return parser.parse_args()


def parse_number(value: str) -> Optional[float]:
    text = str(value).strip().replace(",", "")
    if text in {"", "NA", "NaN", "nan", "."}:
        return None
    return float(text)


def period_key(date_text: str) -> str:
    text = str(date_text).strip()
    if len(text) < 7:
        raise ValueError(f"Expected monthly date string like YYYY-MM-01, got {date_text!r}")
    return text[:7]


def month_to_index(month_key: str) -> int:
    year = int(month_key[:4])
    month = int(month_key[5:7])
    return year * 12 + (month - 1)


def month_sequence(start_month: str, end_month: str) -> List[str]:
    start_idx = month_to_index(start_month)
    end_idx = month_to_index(end_month)
    out = []
    for idx in range(start_idx, end_idx + 1):
        year = idx // 12
        month = (idx % 12) + 1
        out.append(f"{year:04d}-{month:02d}")
    return out


def longest_contiguous_window(months: Iterable[str]) -> Tuple[str, str]:
    ordered = sorted(set(months))
    if not ordered:
        raise ValueError("cannot compute contiguous window for an empty month set")

    best_start = best_end = ordered[0]
    current_start = current_end = ordered[0]
    for month in ordered[1:]:
        if month_to_index(month) == month_to_index(current_end) + 1:
            current_end = month
        else:
            best_len = month_to_index(best_end) - month_to_index(best_start) + 1
            current_len = month_to_index(current_end) - month_to_index(current_start) + 1
            if current_len > best_len or (current_len == best_len and current_end > best_end):
                best_start, best_end = current_start, current_end
            current_start = current_end = month

    best_len = month_to_index(best_end) - month_to_index(best_start) + 1
    current_len = month_to_index(current_end) - month_to_index(current_start) + 1
    if current_len > best_len or (current_len == best_len and current_end > best_end):
        best_start, best_end = current_start, current_end
    return best_start, best_end


def build_tail_masks(length: int, *, val_months: int, test_months: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if val_months <= 0 or test_months <= 0:
        raise ValueError("val-months and test-months must both be positive")

    num_targets = length - 1
    if num_targets <= val_months + test_months:
        raise ValueError(
            f"sequence with {num_targets} one-step targets is too short for val_months={val_months} and test_months={test_months}"
        )

    train_mask = torch.zeros(length, dtype=torch.bool)
    val_mask = torch.zeros(length, dtype=torch.bool)
    test_mask = torch.zeros(length, dtype=torch.bool)
    val_start = length - (val_months + test_months)
    test_start = length - test_months
    train_mask[1:val_start] = True
    val_mask[val_start:test_start] = True
    test_mask[test_start:length] = True
    return train_mask, val_mask, test_mask


def build_fraction_masks(length: int, *, train_fraction: float, val_fraction: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if length < 2:
        raise ValueError(f"sequence length must be at least 2 to form a one-step target split, got {length}")
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
        test_count = num_targets - train_count - val_count
    if min(train_count, val_count, test_count) < 1:
        raise ValueError(
            f"could not form a valid 70/15/15 split from {num_targets} targets: "
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
    return train_mask, val_mask, test_mask


def infer_masks(args: argparse.Namespace, length: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
    num_targets = length - 1
    if args.split_mode == "tail":
        return (*build_tail_masks(length, val_months=args.val_months, test_months=args.test_months), "tail")
    if args.split_mode == "fraction":
        return (*build_fraction_masks(length, train_fraction=args.train_fraction, val_fraction=args.val_fraction), "fraction_70_15_15")
    if num_targets >= args.auto_tail_min_target_months:
        return (*build_tail_masks(length, val_months=args.val_months, test_months=args.test_months), "tail")
    return (*build_fraction_masks(length, train_fraction=args.train_fraction, val_fraction=args.val_fraction), "fraction_70_15_15")


def mask_range(mask: torch.Tensor, months: List[str]) -> Optional[Dict[str, object]]:
    indices = torch.nonzero(mask, as_tuple=False).squeeze(-1)
    if indices.numel() == 0:
        return None
    start_idx = int(indices[0].item())
    end_idx = int(indices[-1].item())
    return {
        "start_month": months[start_idx],
        "end_month": months[end_idx],
        "num_target_steps": int(indices.numel()),
    }


def summarize_lengths(lengths: List[int]) -> Dict[str, float]:
    ordered = sorted(lengths)
    n = len(ordered)
    return {
        "min": float(ordered[0]),
        "median": float(ordered[n // 2]),
        "mean": float(sum(ordered) / n),
        "max": float(ordered[-1]),
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.min_contiguous_months < 24:
        raise ValueError("--min-contiguous-months must be >= 24 for a meaningful monthly benchmark")
    if not input_path.exists():
        raise FileNotFoundError(
            f"missing input file: {input_path}\n"
            "Run benchmarks/ember_monthly/download.py first, or point --input to the official Ember monthly CSV."
        )

    country_info: Dict[CountryKey, Dict[str, object]] = {}
    total_generation: DefaultDict[CountryKey, Dict[str, float]] = defaultdict(dict)
    fuel_generation: DefaultDict[CountryKey, DefaultDict[str, Dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    demand_twh: DefaultDict[CountryKey, Dict[str, float]] = defaultdict(dict)
    net_imports_twh: DefaultDict[CountryKey, Dict[str, float]] = defaultdict(dict)
    total_emissions_mtco2: DefaultDict[CountryKey, Dict[str, float]] = defaultdict(dict)
    co2_intensity_gco2_per_kwh: DefaultDict[CountryKey, Dict[str, float]] = defaultdict(dict)

    with input_path.open("r", encoding=args.encoding, newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{input_path} is missing a CSV header row")

        required_columns = {
            "Area",
            "ISO 3 code",
            "Date",
            "Area type",
            "Continent",
            "Ember region",
            "EU",
            "OECD",
            "G20",
            "G7",
            "ASEAN",
            "Category",
            "Subcategory",
            "Variable",
            "Unit",
            "Value",
        }
        missing = required_columns.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"{input_path} is missing Ember columns: {sorted(missing)}")

        for row in reader:
            if row["Area type"].strip() != "Country or economy":
                continue

            iso = row["ISO 3 code"].strip()
            if not iso or iso.lower() == "nan":
                continue
            month = period_key(row["Date"])
            value = parse_number(row["Value"])
            if value is None:
                continue

            if iso not in country_info:
                country_info[iso] = {
                    "area": row["Area"].strip(),
                    "iso3": iso,
                    "continent": row["Continent"].strip(),
                    "ember_region": row["Ember region"].strip(),
                    "eu": int(float(row["EU"])) if row["EU"].strip() else 0,
                    "oecd": int(float(row["OECD"])) if row["OECD"].strip() else 0,
                    "g20": int(float(row["G20"])) if row["G20"].strip() else 0,
                    "g7": int(float(row["G7"])) if row["G7"].strip() else 0,
                    "asean": int(float(row["ASEAN"])) if row["ASEAN"].strip() else 0,
                }

            category = row["Category"].strip()
            subcategory = row["Subcategory"].strip()
            variable = row["Variable"].strip()
            unit = row["Unit"].strip()

            if category == "Electricity generation" and unit == "TWh":
                if subcategory == "Total" and variable == "Total Generation":
                    if value > 0:
                        total_generation[iso][month] = value
                elif subcategory == "Fuel" and variable in FUEL_VOCAB:
                    fuel_generation[iso][month][variable] = fuel_generation[iso][month].get(variable, 0.0) + value
                continue

            if category == "Electricity demand" and variable == "Demand" and unit == "TWh":
                demand_twh[iso][month] = value
                continue
            if category == "Electricity imports" and variable == "Net Imports" and unit == "TWh":
                net_imports_twh[iso][month] = value
                continue
            if category == "Power sector emissions" and variable == "Total emissions" and unit == "mtCO2":
                total_emissions_mtco2[iso][month] = value
                continue
            if category == "Power sector emissions" and variable == "CO2 intensity" and unit == "gCO2/kWh":
                co2_intensity_gco2_per_kwh[iso][month] = value

    selected_countries = []
    rows = []
    sequence_lengths: List[int] = []
    split_target_totals = Counter()
    split_mode_counts = Counter()
    sum_fuel_twh = 0.0
    sum_total_generation_twh = 0.0
    global_min_month = None
    global_max_month = None

    country_payloads = []
    for iso in sorted(total_generation):
        months = total_generation[iso]
        if not months:
            continue
        start_month, end_month = longest_contiguous_window(months.keys())
        timeline = month_sequence(start_month, end_month)
        if len(timeline) < args.min_contiguous_months:
            continue

        train_mask, val_mask, test_mask, resolved_split_mode = infer_masks(args, len(timeline))
        split_mode_counts[resolved_split_mode] += 1
        train_range = mask_range(train_mask, timeline)
        val_range = mask_range(val_mask, timeline)
        test_range = mask_range(test_mask, timeline)

        sequence = torch.zeros((len(timeline), len(FUEL_VOCAB)), dtype=torch.float32)
        total_generation_tensor = torch.zeros(len(timeline), dtype=torch.float32)
        demand_tensor = torch.zeros(len(timeline), dtype=torch.float32)
        demand_mask = torch.zeros(len(timeline), dtype=torch.bool)
        imports_tensor = torch.zeros(len(timeline), dtype=torch.float32)
        imports_mask = torch.zeros(len(timeline), dtype=torch.bool)
        emissions_tensor = torch.zeros(len(timeline), dtype=torch.float32)
        emissions_mask = torch.zeros(len(timeline), dtype=torch.bool)
        intensity_tensor = torch.zeros(len(timeline), dtype=torch.float32)
        intensity_mask = torch.zeros(len(timeline), dtype=torch.bool)
        month_index_tensor = torch.zeros(len(timeline), dtype=torch.int32)

        for offset, month in enumerate(timeline):
            total_value = float(total_generation[iso].get(month, 0.0))
            if total_value <= 0:
                raise ValueError(f"missing positive total generation for iso={iso} month={month}")
            detailed = fuel_generation[iso].get(month, {})
            fuel_values = [float(detailed.get(fuel, 0.0)) for fuel in FUEL_VOCAB]
            row_sum = sum(fuel_values)
            if row_sum <= 0:
                raise ValueError(f"no positive Ember fuel generation found for iso={iso} month={month}")

            sequence[offset] = torch.tensor(fuel_values, dtype=torch.float32) / float(row_sum)
            total_generation_tensor[offset] = total_value
            if month in demand_twh[iso]:
                demand_tensor[offset] = float(demand_twh[iso][month])
                demand_mask[offset] = True
            if month in net_imports_twh[iso]:
                imports_tensor[offset] = float(net_imports_twh[iso][month])
                imports_mask[offset] = True
            if month in total_emissions_mtco2[iso]:
                emissions_tensor[offset] = float(total_emissions_mtco2[iso][month])
                emissions_mask[offset] = True
            if month in co2_intensity_gco2_per_kwh[iso]:
                intensity_tensor[offset] = float(co2_intensity_gco2_per_kwh[iso][month])
                intensity_mask[offset] = True
            month_index_tensor[offset] = month_to_index(month)
            sum_fuel_twh += row_sum
            sum_total_generation_twh += total_value

        info = country_info.get(iso, {"area": iso, "iso3": iso, "continent": "", "ember_region": "", "eu": 0, "oecd": 0, "g20": 0, "g7": 0, "asean": 0})
        rows.append(
            {
                "sequence_index": len(selected_countries),
                "area": info["area"],
                "iso3": iso,
                "continent": info["continent"],
                "ember_region": info["ember_region"],
                "eu": info["eu"],
                "oecd": info["oecd"],
                "g20": info["g20"],
                "g7": info["g7"],
                "asean": info["asean"],
                "start_month": start_month,
                "end_month": end_month,
                "sequence_length_months": len(timeline),
                "resolved_split_mode": resolved_split_mode,
                "train_target_start_month": train_range["start_month"] if train_range else "",
                "train_target_end_month": train_range["end_month"] if train_range else "",
                "train_target_steps": train_range["num_target_steps"] if train_range else 0,
                "val_target_start_month": val_range["start_month"] if val_range else "",
                "val_target_end_month": val_range["end_month"] if val_range else "",
                "val_target_steps": val_range["num_target_steps"] if val_range else 0,
                "test_target_start_month": test_range["start_month"] if test_range else "",
                "test_target_end_month": test_range["end_month"] if test_range else "",
                "test_target_steps": test_range["num_target_steps"] if test_range else 0,
            }
        )
        split_target_totals["train"] += train_range["num_target_steps"] if train_range else 0
        split_target_totals["val"] += val_range["num_target_steps"] if val_range else 0
        split_target_totals["test"] += test_range["num_target_steps"] if test_range else 0
        sequence_lengths.append(len(timeline))
        selected_countries.append(iso)
        global_min_month = start_month if global_min_month is None else min(global_min_month, start_month)
        global_max_month = end_month if global_max_month is None else max(global_max_month, end_month)
        country_payloads.append(
            (
                sequence,
                train_mask,
                val_mask,
                test_mask,
                total_generation_tensor,
                demand_tensor,
                demand_mask,
                imports_tensor,
                imports_mask,
                emissions_tensor,
                emissions_mask,
                intensity_tensor,
                intensity_mask,
                month_index_tensor,
            )
        )

    if not country_payloads:
        raise ValueError(
            f"No Ember monthly country sequences passed min_contiguous_months >= {args.min_contiguous_months}."
        )

    max_length = max(sequence_lengths)
    num_countries = len(country_payloads)
    inputs = torch.zeros((num_countries, max_length, len(FUEL_VOCAB)), dtype=torch.float32)
    train_loss_mask = torch.zeros((num_countries, max_length), dtype=torch.bool)
    val_loss_mask = torch.zeros((num_countries, max_length), dtype=torch.bool)
    test_loss_mask = torch.zeros((num_countries, max_length), dtype=torch.bool)
    total_generation_tensor = torch.zeros((num_countries, max_length), dtype=torch.float32)
    demand_tensor = torch.zeros((num_countries, max_length), dtype=torch.float32)
    demand_mask = torch.zeros((num_countries, max_length), dtype=torch.bool)
    imports_tensor = torch.zeros((num_countries, max_length), dtype=torch.float32)
    imports_mask = torch.zeros((num_countries, max_length), dtype=torch.bool)
    emissions_tensor = torch.zeros((num_countries, max_length), dtype=torch.float32)
    emissions_mask = torch.zeros((num_countries, max_length), dtype=torch.bool)
    intensity_tensor = torch.zeros((num_countries, max_length), dtype=torch.float32)
    intensity_mask = torch.zeros((num_countries, max_length), dtype=torch.bool)
    month_index = torch.zeros((num_countries, max_length), dtype=torch.int32)
    lengths = torch.tensor(sequence_lengths, dtype=torch.long)

    for idx, payload in enumerate(country_payloads):
        (
            sequence,
            train_mask,
            val_mask,
            test_mask,
            total_generation_seq,
            demand_seq,
            demand_seq_mask,
            imports_seq,
            imports_seq_mask,
            emissions_seq,
            emissions_seq_mask,
            intensity_seq,
            intensity_seq_mask,
            month_index_seq,
        ) = payload
        length = int(sequence.size(0))
        inputs[idx, :length] = sequence
        train_loss_mask[idx, :length] = train_mask
        val_loss_mask[idx, :length] = val_mask
        test_loss_mask[idx, :length] = test_mask
        total_generation_tensor[idx, :length] = total_generation_seq
        demand_tensor[idx, :length] = demand_seq
        demand_mask[idx, :length] = demand_seq_mask
        imports_tensor[idx, :length] = imports_seq
        imports_mask[idx, :length] = imports_seq_mask
        emissions_tensor[idx, :length] = emissions_seq
        emissions_mask[idx, :length] = emissions_seq_mask
        intensity_tensor[idx, :length] = intensity_seq
        intensity_mask[idx, :length] = intensity_seq_mask
        month_index[idx, :length] = month_index_seq

    payload_common = {
        "inputs": inputs,
        "lengths": lengths,
        "total_generation_twh": total_generation_tensor,
        "demand_twh": demand_tensor,
        "demand_mask": demand_mask,
        "net_imports_twh": imports_tensor,
        "net_imports_mask": imports_mask,
        "total_emissions_mtco2": emissions_tensor,
        "total_emissions_mask": emissions_mask,
        "co2_intensity_gco2_per_kwh": intensity_tensor,
        "co2_intensity_mask": intensity_mask,
        "month_index": month_index,
    }
    torch.save({**payload_common, "loss_mask": train_loss_mask}, output_dir / "train.pt")
    torch.save({**payload_common, "loss_mask": val_loss_mask}, output_dir / "val.pt")
    torch.save({**payload_common, "loss_mask": test_loss_mask}, output_dir / "test.pt")

    vocab_csv_path = output_dir / "vocab.csv"
    with vocab_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["index", "variable", "category"])
        writer.writeheader()
        for idx, variable in enumerate(FUEL_VOCAB):
            writer.writerow({"index": idx, "variable": variable, "category": "Electricity generation / Fuel / TWh"})

    geographies_csv_path = output_dir / "geographies.csv"
    with geographies_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "benchmark": TRACK_NAME,
        "description": (
            "Monthly country-level electricity generation-mix benchmark from Ember's official long-format release, "
            "using the nine detailed fuel generation variables that sum to total generation."
        ),
        "source_input": str(input_path),
        "num_countries": num_countries,
        "num_sequences": num_countries,
        "probability_dim": len(FUEL_VOCAB),
        "fuel_vocabulary": FUEL_VOCAB,
        "time_unit": "month",
        "sequence_unit": "country_or_economy",
        "selected_iso3": selected_countries,
        "global_start_month": global_min_month,
        "global_end_month": global_max_month,
        "sequence_length_months_summary": summarize_lengths(sequence_lengths),
        "split_mode_requested": args.split_mode,
        "split_mode_counts": dict(split_mode_counts),
        "split_target_totals": dict(split_target_totals),
        "val_months": args.val_months,
        "test_months": args.test_months,
        "auto_tail_min_target_months": args.auto_tail_min_target_months,
        "min_contiguous_months": args.min_contiguous_months,
        "train_fraction": args.train_fraction,
        "val_fraction": args.val_fraction,
        "fuel_sum_matches_total_generation": abs(sum_fuel_twh - sum_total_generation_twh) <= 1e-6 * max(1.0, sum_total_generation_twh),
        "sum_detailed_fuel_twh": sum_fuel_twh,
        "sum_total_generation_twh": sum_total_generation_twh,
        "split_rationale": (
            "Use the longest contiguous monthly country window to avoid missing-month artifacts, "
            "reserve the final 24 target months as 12-month validation + 12-month test windows when the sequence is long enough, "
            "and fall back to a chronological 70/15/15 split for shorter country histories."
        ),
        "artifacts": {
            "vocab_csv": str(vocab_csv_path),
            "geographies_csv": str(geographies_csv_path),
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"[prepare] wrote {output_dir}")
    print(
        f"[prepare] countries={num_countries} dim={len(FUEL_VOCAB)} "
        f"global_window={global_min_month}..{global_max_month}"
    )
    print(
        f"[prepare] sequence_length_months min={metadata['sequence_length_months_summary']['min']:.0f} "
        f"median={metadata['sequence_length_months_summary']['median']:.0f} "
        f"max={metadata['sequence_length_months_summary']['max']:.0f}"
    )
    print(
        f"[prepare] split_target_totals train={split_target_totals['train']} "
        f"val={split_target_totals['val']} test={split_target_totals['test']}"
    )


if __name__ == "__main__":
    main()
