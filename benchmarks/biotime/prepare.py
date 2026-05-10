from __future__ import annotations

import argparse
import csv
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, Iterator, List, Optional, Tuple

import torch

TRACK_NAME = "biotime_birds_species_annual"
OTHER_TOKEN = "OTHER"
DEFAULT_TAXON = "Birds"
DEFAULT_TOP_K = 127
DEFAULT_MIN_CONTIGUOUS_YEARS = 15
DEFAULT_COORDINATE_DECIMALS = 6
SiteKey = Tuple[str, str, str, str]


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parent
    raw_dir = root_dir / "raw"
    parser = argparse.ArgumentParser(
        description="Prepare a fixed-vocabulary BioTIME composition benchmark in tensor_dir format."
    )
    parser.add_argument(
        "--input",
        default=str(raw_dir / "biotime_raw.zip"),
        help="Path to the BioTIME raw observation zip or CSV.",
    )
    parser.add_argument(
        "--metadata",
        default=str(raw_dir / "biotime_metadata.csv"),
        help="Path to the BioTIME study metadata CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(root_dir / "processed"),
        help="Directory where train.pt / val.pt / test.pt and metadata.json will be written.",
    )
    parser.add_argument(
        "--taxon",
        default=DEFAULT_TAXON,
        help="Taxon subset to retain. Defaults to Birds for the current benchmark track.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of global species tokens to keep before adding OTHER.",
    )
    parser.add_argument(
        "--min-contiguous-years",
        type=int,
        default=DEFAULT_MIN_CONTIGUOUS_YEARS,
        help="Minimum length of the longest contiguous annual site window kept in the benchmark.",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.70,
        help="Chronological train fraction applied to one-step target years within each retained sequence.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.15,
        help="Chronological validation fraction applied to one-step target years within each retained sequence.",
    )
    parser.add_argument(
        "--coordinate-decimals",
        type=int,
        default=DEFAULT_COORDINATE_DECIMALS,
        help="Decimals used when canonicalizing latitude / longitude / depth site keys.",
    )
    parser.add_argument(
        "--include-non-species",
        action="store_true",
        help="Allow mixed taxonomic resolution rows. By default only species-level rows are kept.",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8-sig",
        help="Text encoding for CSV reads. Defaults to utf-8-sig.",
    )
    return parser.parse_args()


def parse_optional_float(value: str) -> Optional[float]:
    text = str(value).strip()
    if text in {"", "NA", "NaN", "nan", "."}:
        return None
    return float(text)


def parse_positive_measure(row: Dict[str, str]) -> Tuple[Optional[float], Optional[str]]:
    abundance = parse_optional_float(row.get("ABUNDANCE", ""))
    if abundance is not None and abundance > 0:
        return abundance, "abundance"

    biomass = parse_optional_float(row.get("BIOMASS", ""))
    if biomass is not None and biomass > 0:
        return biomass, "biomass"

    return None, None


def canonical_numeric(value: str, *, decimals: int) -> str:
    parsed = parse_optional_float(value)
    if parsed is None:
        return "NA"
    return f"{parsed:.{decimals}f}"


def canonical_site_key(row: Dict[str, str], *, decimals: int) -> SiteKey:
    return (
        row["STUDY_ID"].strip(),
        canonical_numeric(row.get("LATITUDE", ""), decimals=decimals),
        canonical_numeric(row.get("LONGITUDE", ""), decimals=decimals),
        canonical_numeric(row.get("DEPTH", ""), decimals=decimals),
    )


def iter_observation_rows(input_path: Path, *, encoding: str) -> Iterator[Dict[str, str]]:
    if input_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(input_path) as zf:
            for member in sorted(zf.namelist()):
                if not member.lower().endswith(".csv"):
                    continue
                with zf.open(member) as handle:
                    text_handle = (line.decode(encoding, errors="replace") for line in handle)
                    reader = csv.DictReader(text_handle)
                    if reader.fieldnames is None:
                        raise ValueError(f"{input_path} member {member} is missing a CSV header row")
                    required = {
                        "ABUNDANCE",
                        "BIOMASS",
                        "LATITUDE",
                        "LONGITUDE",
                        "DEPTH",
                        "YEAR",
                        "STUDY_ID",
                        "newID",
                        "valid_name",
                        "resolution",
                        "taxon",
                    }
                    missing = required.difference(reader.fieldnames)
                    if missing:
                        raise ValueError(f"{input_path} member {member} is missing columns: {sorted(missing)}")
                    for row in reader:
                        yield row
        return

    with input_path.open("r", encoding=encoding, newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{input_path} is missing a CSV header row")
        required = {
            "ABUNDANCE",
            "BIOMASS",
            "LATITUDE",
            "LONGITUDE",
            "DEPTH",
            "YEAR",
            "STUDY_ID",
            "newID",
            "valid_name",
            "resolution",
            "taxon",
        }
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"{input_path} is missing columns: {sorted(missing)}")
        for row in reader:
            yield row


def load_study_metadata(metadata_path: Path, *, encoding: str) -> Dict[str, Dict[str, str]]:
    if not metadata_path.exists():
        print(f"[prepare] metadata file not found; continuing without study metadata: {metadata_path}")
        return {}

    with metadata_path.open("r", encoding=encoding, newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{metadata_path} is missing a CSV header row")
        if "STUDY_ID" not in reader.fieldnames:
            raise ValueError(f"{metadata_path} is missing required column STUDY_ID")
        return {row["STUDY_ID"].strip(): row for row in reader if row.get("STUDY_ID", "").strip()}


def keep_row(row: Dict[str, str], args: argparse.Namespace) -> Tuple[bool, Optional[int], Optional[SiteKey], Optional[float], Optional[str]]:
    if row.get("taxon", "").strip() != args.taxon:
        return False, None, None, None, None
    if not args.include_non_species and row.get("resolution", "").strip().lower() != "species":
        return False, None, None, None, None

    measure_value, measure_source = parse_positive_measure(row)
    if measure_value is None:
        return False, None, None, None, None

    try:
        year = int(float(row["YEAR"]))
    except Exception:
        return False, None, None, None, None

    site = canonical_site_key(row, decimals=args.coordinate_decimals)
    return True, year, site, measure_value, measure_source


def longest_contiguous_window(years: Iterable[int]) -> Tuple[int, int]:
    ordered = sorted(set(years))
    if not ordered:
        raise ValueError("cannot compute contiguous window for an empty year set")

    best_start = best_end = ordered[0]
    current_start = current_end = ordered[0]
    for year in ordered[1:]:
        if year == current_end + 1:
            current_end = year
        else:
            best_len = best_end - best_start + 1
            current_len = current_end - current_start + 1
            if current_len > best_len or (current_len == best_len and current_end > best_end):
                best_start, best_end = current_start, current_end
            current_start = current_end = year

    best_len = best_end - best_start + 1
    current_len = current_end - current_start + 1
    if current_len > best_len or (current_len == best_len and current_end > best_end):
        best_start, best_end = current_start, current_end
    return best_start, best_end


def build_fraction_masks(length: int, *, train_fraction: float, val_fraction: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[int, int, int]]:
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
            deficit -= val_reduction
        test_count = num_targets - train_count - val_count

    if min(train_count, val_count, test_count) < 1:
        raise ValueError(
            f"could not form a valid per-sequence split from {num_targets} targets: "
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


def summarize_lengths(lengths: List[int]) -> Dict[str, float]:
    ordered = sorted(lengths)
    n = len(ordered)
    return {
        "min": float(ordered[0]),
        "median": float(ordered[n // 2]),
        "mean": float(sum(ordered) / n),
        "max": float(ordered[-1]),
    }


def site_sort_key(site: SiteKey) -> Tuple[int, str, str, str]:
    study_id = site[0]
    numeric_study = int(study_id) if study_id.isdigit() else 10**9
    return numeric_study, site[1], site[2], site[3]


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


def write_vocab_csv(path: Path, vocab_ids: List[str], species_names: Dict[str, str], species_total: Counter[str], total_mass: float) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["index", "token_type", "species_id", "species_name", "total_measure", "measure_share"],
        )
        writer.writeheader()
        for idx, species_id in enumerate(vocab_ids):
            if species_id == OTHER_TOKEN:
                writer.writerow(
                    {
                        "index": idx,
                        "token_type": "other",
                        "species_id": OTHER_TOKEN,
                        "species_name": OTHER_TOKEN,
                        "total_measure": "",
                        "measure_share": "",
                    }
                )
                continue
            measure = float(species_total[species_id])
            writer.writerow(
                {
                    "index": idx,
                    "token_type": "species",
                    "species_id": species_id,
                    "species_name": species_names.get(species_id, species_id),
                    "total_measure": measure,
                    "measure_share": measure / total_mass if total_mass > 0 else 0.0,
                }
            )


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    metadata_path = Path(args.metadata).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.top_k < 1:
        raise ValueError("--top-k must be >= 1")
    if args.min_contiguous_years < 4:
        raise ValueError("--min-contiguous-years must be >= 4 for a meaningful train/val/test split")
    if not input_path.exists():
        raise FileNotFoundError(
            f"missing input file: {input_path}\n"
            "Run benchmarks/biotime/download.py first, or point --input to the official BioTIME raw zip / CSV."
        )

    study_metadata = load_study_metadata(metadata_path, encoding=args.encoding)

    site_years: DefaultDict[SiteKey, set[int]] = defaultdict(set)
    first_pass_measure_source = Counter()
    for row in iter_observation_rows(input_path, encoding=args.encoding):
        keep, year, site, _value, measure_source = keep_row(row, args)
        if not keep or year is None or site is None or measure_source is None:
            continue
        site_years[site].add(year)
        first_pass_measure_source[measure_source] += 1

    eligible_windows: Dict[SiteKey, Tuple[int, int]] = {}
    for site, years in site_years.items():
        start_year, end_year = longest_contiguous_window(years)
        if end_year - start_year + 1 >= args.min_contiguous_years:
            eligible_windows[site] = (start_year, end_year)

    if not eligible_windows:
        raise ValueError(
            f"No BioTIME sites passed the longest contiguous window threshold >= {args.min_contiguous_years} years "
            f"for taxon={args.taxon!r}."
        )

    site_year_species: DefaultDict[SiteKey, DefaultDict[int, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    site_year_total_measure: DefaultDict[SiteKey, Dict[int, float]] = defaultdict(dict)
    species_total_measure: Counter[str] = Counter()
    species_names: Dict[str, str] = {}
    second_pass_measure_source = Counter()

    for row in iter_observation_rows(input_path, encoding=args.encoding):
        keep, year, site, measure_value, measure_source = keep_row(row, args)
        if not keep or year is None or site is None or measure_value is None or measure_source is None:
            continue

        window = eligible_windows.get(site)
        if window is None or year < window[0] or year > window[1]:
            continue

        species_id = row.get("newID", "").strip() or row.get("valid_name", "").strip()
        if not species_id:
            continue

        site_year_species[site][year][species_id] += measure_value
        site_year_total_measure[site][year] = site_year_total_measure[site].get(year, 0.0) + measure_value
        species_total_measure[species_id] += measure_value
        species_names.setdefault(species_id, row.get("valid_name", "").strip() or species_id)
        second_pass_measure_source[measure_source] += measure_value

    if not species_total_measure:
        raise ValueError(
            "No retained BioTIME species mass remained after filtering. "
            "Try lowering --min-contiguous-years or changing --taxon / --include-non-species."
        )

    top_species = [species_id for species_id, _ in species_total_measure.most_common(args.top_k)]
    vocab_ids = top_species + [OTHER_TOKEN]
    vocab_index = {species_id: idx for idx, species_id in enumerate(vocab_ids)}
    captured_mass = sum(species_total_measure[species_id] for species_id in top_species)
    total_mass = float(sum(species_total_measure.values()))

    selected_sites = sorted(eligible_windows, key=site_sort_key)
    sequence_lengths: List[int] = []
    split_target_totals = Counter()
    study_ids = sorted(
        {site[0] for site in selected_sites},
        key=lambda value: (0, int(value)) if value.isdigit() else (1, value),
    )

    site_rows: List[Dict[str, object]] = []
    sequence_payloads: List[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, SiteKey]
    ] = []
    for sequence_index, site in enumerate(selected_sites):
        start_year, end_year = eligible_windows[site]
        years = list(range(start_year, end_year + 1))
        length = len(years)
        train_mask, val_mask, test_mask, split_counts = build_fraction_masks(
            length,
            train_fraction=args.train_fraction,
            val_fraction=args.val_fraction,
        )
        split_target_totals["train"] += split_counts[0]
        split_target_totals["val"] += split_counts[1]
        split_target_totals["test"] += split_counts[2]

        sequence = torch.zeros((length, len(vocab_ids)), dtype=torch.float32)
        total_measure_tensor = torch.zeros(length, dtype=torch.float32)
        richness_tensor = torch.zeros(length, dtype=torch.int32)
        year_tensor = torch.tensor(years, dtype=torch.int32)

        for offset, year in enumerate(years):
            by_species = site_year_species[site].get(year)
            total_measure = float(site_year_total_measure[site].get(year, 0.0))
            if by_species is None or total_measure <= 0:
                raise ValueError(f"missing aggregated BioTIME measure for site={site} year={year}")

            selected_measure = 0.0
            for species_id, measure in by_species.items():
                vocab_slot = vocab_index.get(species_id)
                if vocab_slot is None or species_id == OTHER_TOKEN:
                    continue
                sequence[offset, vocab_slot] += float(measure)
                selected_measure += float(measure)

            other_measure = max(total_measure - selected_measure, 0.0)
            sequence[offset, vocab_index[OTHER_TOKEN]] = other_measure
            sequence[offset] /= float(sequence[offset].sum().item())
            total_measure_tensor[offset] = total_measure
            richness_tensor[offset] = len(by_species)

        train_range = year_range_from_mask(train_mask, years)
        val_range = year_range_from_mask(val_mask, years)
        test_range = year_range_from_mask(test_mask, years)
        study_info = study_metadata.get(site[0], {})
        site_rows.append(
            {
                "sequence_index": sequence_index,
                "study_id": site[0],
                "site_id": f"{site[0]}|{site[1]}|{site[2]}|{site[3]}",
                "latitude": site[1],
                "longitude": site[2],
                "depth": site[3],
                "start_year": start_year,
                "end_year": end_year,
                "sequence_length_years": length,
                "train_targets": split_counts[0],
                "val_targets": split_counts[1],
                "test_targets": split_counts[2],
                "train_target_start_year": train_range["start_year"] if train_range else "",
                "train_target_end_year": train_range["end_year"] if train_range else "",
                "val_target_start_year": val_range["start_year"] if val_range else "",
                "val_target_end_year": val_range["end_year"] if val_range else "",
                "test_target_start_year": test_range["start_year"] if test_range else "",
                "test_target_end_year": test_range["end_year"] if test_range else "",
                "realm": study_info.get("REALM", ""),
                "climate": study_info.get("CLIMATE", ""),
                "habitat": study_info.get("HABITAT", ""),
                "taxa": study_info.get("TAXA", ""),
                "organisms": study_info.get("ORGANISMS", ""),
                "title": study_info.get("TITLE", ""),
            }
        )
        sequence_lengths.append(length)
        sequence_payloads.append((sequence, train_mask, val_mask, test_mask, total_measure_tensor, richness_tensor, year_tensor, site))

    max_length = max(sequence_lengths)
    num_sequences = len(sequence_payloads)
    inputs = torch.zeros((num_sequences, max_length, len(vocab_ids)), dtype=torch.float32)
    train_loss_mask = torch.zeros((num_sequences, max_length), dtype=torch.bool)
    val_loss_mask = torch.zeros((num_sequences, max_length), dtype=torch.bool)
    test_loss_mask = torch.zeros((num_sequences, max_length), dtype=torch.bool)
    total_measure = torch.zeros((num_sequences, max_length), dtype=torch.float32)
    species_richness = torch.zeros((num_sequences, max_length), dtype=torch.int32)
    observation_year = torch.zeros((num_sequences, max_length), dtype=torch.int32)
    lengths = torch.tensor(sequence_lengths, dtype=torch.long)

    for idx, (sequence, train_mask, val_mask, test_mask, total_measure_tensor, richness_tensor, year_tensor, _site) in enumerate(sequence_payloads):
        length = int(sequence.size(0))
        inputs[idx, :length] = sequence
        train_loss_mask[idx, :length] = train_mask
        val_loss_mask[idx, :length] = val_mask
        test_loss_mask[idx, :length] = test_mask
        total_measure[idx, :length] = total_measure_tensor
        species_richness[idx, :length] = richness_tensor
        observation_year[idx, :length] = year_tensor

    payload_common = {
        "inputs": inputs,
        "lengths": lengths,
        "total_measure": total_measure,
        "species_richness": species_richness,
        "observation_year": observation_year,
    }
    torch.save({**payload_common, "loss_mask": train_loss_mask}, output_dir / "train.pt")
    torch.save({**payload_common, "loss_mask": val_loss_mask}, output_dir / "val.pt")
    torch.save({**payload_common, "loss_mask": test_loss_mask}, output_dir / "test.pt")

    vocab_path = output_dir / "vocab.csv"
    write_vocab_csv(vocab_path, vocab_ids, species_names, species_total_measure, total_mass)

    sites_csv_path = output_dir / "sites.csv"
    with sites_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(site_rows[0].keys()))
        writer.writeheader()
        writer.writerows(site_rows)

    realm_counts = Counter(row["realm"] for row in site_rows if row["realm"])
    metadata = {
        "benchmark": TRACK_NAME,
        "description": (
            "Bird community composition benchmark derived from BioTIME 2.0 site-level annual sequences, "
            "filtered to species-level rows and a global top-K species vocabulary plus OTHER."
        ),
        "source_input": str(input_path),
        "source_metadata": str(metadata_path) if metadata_path.exists() else None,
        "taxon_filter": args.taxon,
        "species_only": not args.include_non_species,
        "top_k": args.top_k,
        "probability_dim": len(vocab_ids),
        "other_token": OTHER_TOKEN,
        "min_contiguous_years": args.min_contiguous_years,
        "coordinate_decimals": args.coordinate_decimals,
        "sequence_unit": "study_id x latitude x longitude x depth",
        "time_unit": "year",
        "split_policy": "per-sequence chronological 70/15/15 on target years",
        "train_fraction": args.train_fraction,
        "val_fraction": args.val_fraction,
        "test_fraction": 1.0 - args.train_fraction - args.val_fraction,
        "num_sequences": num_sequences,
        "num_studies": len(study_ids),
        "num_site_years": int(sum(sequence_lengths)),
        "sequence_length_years_summary": summarize_lengths(sequence_lengths),
        "split_target_totals": dict(split_target_totals),
        "captured_measure_fraction_top_k": captured_mass / total_mass if total_mass > 0 else 0.0,
        "num_distinct_species_retained_before_top_k": len(species_total_measure),
        "selected_study_ids": study_ids,
        "selected_realm_counts": dict(realm_counts),
        "measure_source_mass": dict(second_pass_measure_source),
        "measure_source_row_counts_first_pass": dict(first_pass_measure_source),
        "split_rationale": (
            "Use the longest contiguous annual site window per sequence to avoid irregular-time artifacts, "
            "require at least 15 annual observations so each site has a non-trivial validation and test holdout, "
            "and split chronologically within each retained sequence."
        ),
        "track_rationale": (
            "Restrict the current unified BioTIME benchmark to Birds because a cross-taxon global species vocabulary "
            "would be ecologically incoherent. Keep species-only rows to avoid mixing taxonomic resolutions."
        ),
        "artifacts": {
            "vocab_csv": str(vocab_path),
            "sites_csv": str(sites_csv_path),
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"[prepare] wrote {output_dir}")
    print(
        f"[prepare] sequences={num_sequences} studies={len(study_ids)} "
        f"dim={len(vocab_ids)} captured_top_k_mass={metadata['captured_measure_fraction_top_k']:.4f}"
    )
    print(
        f"[prepare] sequence_length_years min={metadata['sequence_length_years_summary']['min']:.0f} "
        f"median={metadata['sequence_length_years_summary']['median']:.0f} "
        f"max={metadata['sequence_length_years_summary']['max']:.0f}"
    )
    print(
        f"[prepare] split_target_totals train={split_target_totals['train']} "
        f"val={split_target_totals['val']} test={split_target_totals['test']}"
    )


if __name__ == "__main__":
    main()
