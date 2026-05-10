from __future__ import annotations

import argparse
import csv
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional

import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simplex_benchmark_utils import (
    build_fraction_masks,
    infer_split_masks,
    longest_contiguous_window,
    mask_range,
    pad_sequences,
    save_tensor_splits,
    summarize_lengths,
    write_csv,
    write_json,
)

TRACK_NAME = "bls_qcew_state_industry_employment_mix"
SUPERSECTOR_CODES = [
    "101",
    "1011",
    "1012",
    "1013",
    "102",
    "1021",
    "1022",
    "1023",
    "1024",
    "1025",
    "1026",
    "1027",
    "1028",
]


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parent
    raw_dir = root_dir / "raw"
    parser = argparse.ArgumentParser(description="Prepare a BLS QCEW industry employment-share benchmark.")
    parser.add_argument("--input-dir", default=str(raw_dir))
    parser.add_argument("--output-dir", default=str(root_dir / "processed"))
    parser.add_argument("--start-year", type=int, default=2010)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--geography-level", choices=("state", "county"), default="state")
    parser.add_argument("--industry-level", choices=("supersector", "sector", "observed"), default="supersector")
    parser.add_argument(
        "--own-code",
        default="all",
        help="QCEW ownership code to keep. Use 'all' to sum private/federal/state/local ownership rows.",
    )
    parser.add_argument("--sector-agglvl-code", default="50", help="QCEW aggregation level used for sector mode.")
    parser.add_argument("--min-years", type=int, default=12)
    parser.add_argument("--split-mode", choices=("auto", "tail", "fraction"), default="fraction")
    parser.add_argument("--val-years", type=int, default=2)
    parser.add_argument("--test-years", type=int, default=2)
    parser.add_argument("--auto-tail-min-target-years", type=int, default=20)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--encoding", default="utf-8-sig")
    return parser.parse_args()


def read_titles(path: Path, key: str, value: str) -> Dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return {row[key].strip(): row[value].strip() for row in reader if row.get(key)}


def annual_zip_path(input_dir: Path, year: int) -> Path:
    return input_dir / f"{year}_annual_singlefile.zip"


def csv_name_in_zip(zf: zipfile.ZipFile) -> str:
    names = [name for name in zf.namelist() if name.lower().endswith(".csv")]
    if not names:
        raise ValueError("QCEW zip did not contain a CSV file")
    return names[0]


def is_state_area(area_fips: str) -> bool:
    return len(area_fips) == 5 and area_fips[:2].isdigit() and area_fips[2:] == "000" and area_fips != "00000"


def is_county_area(area_fips: str) -> bool:
    return len(area_fips) == 5 and area_fips.isdigit() and area_fips[2:] != "000"


def keep_area(area_fips: str, level: str) -> bool:
    return is_state_area(area_fips) if level == "state" else is_county_area(area_fips)


def keep_industry(row: Dict[str, str], args: argparse.Namespace) -> bool:
    industry_code = row["industry_code"].strip()
    if args.industry_level == "supersector":
        return industry_code in SUPERSECTOR_CODES
    if args.industry_level == "sector":
        return row["agglvl_code"].strip() == args.sector_agglvl_code and industry_code != "10"
    return industry_code != "10"


def keep_ownership(own_code: str, requested: str) -> bool:
    if requested == "all":
        return own_code in {"1", "2", "3", "5"}
    return own_code == requested


def parse_employment(row: Dict[str, str]) -> Optional[float]:
    raw = row.get("annual_avg_emplvl") or row.get("avg_annual_emplvl") or ""
    text = raw.strip().replace(",", "")
    if not text or text in {"-", "NA", "NaN"}:
        return None
    value = float(text)
    return value if value >= 0 else None


def year_to_index(year: str) -> int:
    return int(year)


def index_to_year(index: int) -> str:
    return str(index)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    area_titles = read_titles(input_dir / "area_titles.csv", "area_fips", "area_title")
    industry_titles = read_titles(input_dir / "industry_titles.csv", "industry_code", "industry_title")

    counts: DefaultDict[str, DefaultDict[int, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    raw_rows_seen = 0
    retained_rows = 0
    industry_seen = Counter()

    for year in range(args.start_year, args.end_year + 1):
        path = annual_zip_path(input_dir, year)
        if not path.exists():
            raise FileNotFoundError(
                f"missing {path}\nRun benchmarks/bls_qcew/download.py --start-year {args.start_year} --end-year {args.end_year} first."
            )
        with zipfile.ZipFile(path) as zf:
            with zf.open(csv_name_in_zip(zf), "r") as raw_handle:
                text_handle = (line.decode(args.encoding, errors="replace") for line in raw_handle)
                reader = csv.DictReader(text_handle)
                required = {"area_fips", "own_code", "industry_code", "agglvl_code", "year"}
                if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                    raise ValueError(f"{path} is missing required QCEW columns")
                for row in reader:
                    raw_rows_seen += 1
                    if not keep_ownership(row["own_code"].strip(), args.own_code):
                        continue
                    area_fips = row["area_fips"].strip()
                    if not keep_area(area_fips, args.geography_level):
                        continue
                    if not keep_industry(row, args):
                        continue
                    employment = parse_employment(row)
                    if employment is None:
                        continue
                    row_year = int(row["year"])
                    industry_code = row["industry_code"].strip()
                    counts[area_fips][row_year][industry_code] += employment
                    industry_seen[industry_code] += employment
                    retained_rows += 1

    if args.industry_level == "supersector":
        vocab = [code for code in SUPERSECTOR_CODES if industry_seen.get(code, 0.0) > 0]
    else:
        vocab = sorted(industry_seen)
    if not vocab:
        raise ValueError("no QCEW industry rows survived filtering; check --industry-level and --sector-agglvl-code")
    vocab_index = {code: idx for idx, code in enumerate(vocab)}

    sequences: List[torch.Tensor] = []
    train_masks: List[torch.Tensor] = []
    val_masks: List[torch.Tensor] = []
    test_masks: List[torch.Tensor] = []
    entity_rows: List[dict[str, object]] = []
    split_mode_counts = Counter()
    split_target_totals = Counter()

    for area_fips in sorted(counts):
        year_map = counts[area_fips]
        valid_years = []
        vectors: Dict[int, torch.Tensor] = {}
        for year, industry_counts in year_map.items():
            vector = torch.zeros(len(vocab), dtype=torch.float32)
            for code, value in industry_counts.items():
                idx = vocab_index.get(code)
                if idx is not None:
                    vector[idx] += float(value)
            total = float(vector.sum().item())
            if total <= 0:
                continue
            vectors[year] = vector / total
            valid_years.append(str(year))
        if not valid_years:
            continue

        start_year, end_year = longest_contiguous_window(
            valid_years,
            to_index=year_to_index,
            from_index=index_to_year,
        )
        years = [str(year) for year in range(int(start_year), int(end_year) + 1) if int(year) in vectors]
        if len(years) < args.min_years:
            continue
        if len(years) != int(end_year) - int(start_year) + 1:
            continue

        sequence = torch.stack([vectors[int(year)] for year in years], dim=0)
        try:
            train_mask, val_mask, test_mask, split_mode = infer_split_masks(
                len(years),
                split_mode=args.split_mode,
                train_fraction=args.train_fraction,
                val_fraction=args.val_fraction,
                val_steps=args.val_years,
                test_steps=args.test_years,
                auto_tail_min_targets=args.auto_tail_min_target_years,
            )
        except ValueError:
            train_mask, val_mask, test_mask = build_fraction_masks(
                len(years),
                train_fraction=args.train_fraction,
                val_fraction=args.val_fraction,
            )
            split_mode = "fraction_70_15_15"

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
                "entity_id": area_fips,
                "entity_label": area_titles.get(area_fips, area_fips),
                "area_fips": area_fips,
                "start_year": years[0],
                "end_year": years[-1],
                "length": len(years),
                "train_start": (mask_range(train_mask, years) or {}).get("start", ""),
                "train_end": (mask_range(train_mask, years) or {}).get("end", ""),
                "val_start": (mask_range(val_mask, years) or {}).get("start", ""),
                "val_end": (mask_range(val_mask, years) or {}).get("end", ""),
                "test_start": (mask_range(test_mask, years) or {}).get("start", ""),
                "test_end": (mask_range(test_mask, years) or {}).get("end", ""),
            }
        )

    if not sequences:
        raise ValueError("no QCEW entity sequences survived filtering")

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
            {
                "index": idx,
                "industry_code": code,
                "token_label": industry_titles.get(code, code),
                "captured_employment": float(industry_seen.get(code, 0.0)),
            }
            for idx, code in enumerate(vocab)
        ],
        ("index", "industry_code", "token_label", "captured_employment"),
    )
    write_csv(
        output_dir / "entities.csv",
        entity_rows,
        (
            "entity_id",
            "entity_label",
            "area_fips",
            "start_year",
            "end_year",
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
        "description": "BLS QCEW state industry-employment share benchmark.",
        "source": "BLS QCEW annual singlefile public CSV zips",
        "source_url_template": "https://data.bls.gov/cew/data/files/{year}/csv/{year}_annual_singlefile.zip",
        "access_status": "public direct access; no application or registration required",
        "geography_level": args.geography_level,
        "industry_level": args.industry_level,
        "own_code": args.own_code,
        "start_year_requested": args.start_year,
        "end_year_requested": args.end_year,
        "num_sequences": len(sequences),
        "probability_dim": len(vocab),
        "sequence_length_years_summary": summarize_lengths(lengths.tolist()),
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
    print(f"[prepare] wrote {output_dir} sequences={len(sequences)} dim={len(vocab)}")


if __name__ == "__main__":
    main()
