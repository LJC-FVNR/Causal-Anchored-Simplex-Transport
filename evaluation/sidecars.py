from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence
from datetime import date, timedelta


def _read_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def infer_token_labels(dataset_dir: Path, prob_dim: int) -> List[str]:
    vocab_path = dataset_dir / "vocab.csv"
    if vocab_path.exists():
        rows = _read_csv_rows(vocab_path)
        for key in ("age_label", "group_label", "item_label", "variable", "token_label", "species_name", "age"):
            if rows and key in rows[0]:
                labels = [row.get(key, "").strip() or f"token_{idx}" for idx, row in enumerate(rows)]
                if len(labels) == prob_dim:
                    return labels
        if rows:
            labels = []
            for idx, row in enumerate(rows):
                value = next((v for k, v in row.items() if k != "index" and v.strip()), "")
                labels.append(value or f"token_{idx}")
            if len(labels) == prob_dim:
                return labels

    metadata = _read_json(dataset_dir / "metadata.json")
    for key in ("age_bins", "food_columns"):
        values = metadata.get(key)
        if isinstance(values, list) and len(values) == prob_dim:
            return [str(value) for value in values]

    return [f"token_{idx}" for idx in range(prob_dim)]


def infer_sequence_labels(dataset_dir: Path, num_sequences: int) -> List[str]:
    metadata = _read_json(dataset_dir / "metadata.json")
    locations = metadata.get("locations")
    if isinstance(locations, list) and len(locations) == num_sequences:
        labels = []
        for idx, item in enumerate(locations):
            if isinstance(item, dict):
                label = item.get("name") or item.get("iso3") or item.get("loc_id")
                labels.append(str(label) if label is not None else f"sequence_{idx}")
            else:
                labels.append(str(item))
        return labels

    for filename, fields in (
        ("entities.csv", ("entity_label", "entity_id")),
        ("countries.csv", ("country_name_en",)),
        ("states.csv", ("state",)),
        ("populations.csv", ("population_code",)),
        ("geographies.csv", ("area", "iso3")),
        ("sites.csv", ("site_id", "title")),
    ):
        rows = _read_csv_rows(dataset_dir / filename)
        if len(rows) != num_sequences:
            continue
        labels = []
        for idx, row in enumerate(rows):
            pieces = [row.get(field, "").strip() for field in fields if row.get(field, "").strip()]
            labels.append(" | ".join(pieces) if pieces else f"sequence_{idx}")
        return labels

    return [f"sequence_{idx}" for idx in range(num_sequences)]


def _month_range(start_month: str, count: int) -> List[str]:
    year_str, month_str = start_month.split("-", 1)
    year = int(year_str)
    month = int(month_str)
    labels = []
    for offset in range(1, count + 1):
        month_idx = month - 1 + offset
        out_year = year + month_idx // 12
        out_month = month_idx % 12 + 1
        labels.append(f"{out_year:04d}-{out_month:02d}")
    return labels


def _year_range(start_year: str, count: int) -> List[str]:
    year = int(start_year)
    return [str(year + step) for step in range(1, count + 1)]


def _week_range(start_week: str, count: int) -> List[str]:
    current = date.fromisoformat(start_week)
    labels = []
    for _ in range(count):
        current = current + timedelta(days=7)
        labels.append(current.isoformat())
    return labels


def infer_time_labels(dataset_dir: Path, sequence_index: int, target_length: int) -> List[str]:
    metadata = _read_json(dataset_dir / "metadata.json")
    years = metadata.get("years")
    if isinstance(years, list) and len(years) >= target_length + 1:
        return [str(value) for value in years[1 : target_length + 1]]

    periods = metadata.get("periods")
    if isinstance(periods, list) and len(periods) >= target_length + 1:
        return [str(value) for value in periods[1 : target_length + 1]]

    week_rows = _read_csv_rows(dataset_dir / "weeks.csv")
    if len(week_rows) >= target_length + 1 and {"mmwr_year", "mmwr_week"} <= set(week_rows[0]):
        out = []
        for row in week_rows[1 : target_length + 1]:
            out.append(f"{int(row['mmwr_year']):04d}W{int(row['mmwr_week']):02d}")
        return out

    for filename in ("countries.csv", "populations.csv", "sites.csv"):
        rows = _read_csv_rows(dataset_dir / filename)
        if sequence_index < len(rows) and "start_year" in rows[sequence_index]:
            start_year = int(rows[sequence_index]["start_year"])
            return [str(start_year + step) for step in range(1, target_length + 1)]

    geography_rows = _read_csv_rows(dataset_dir / "geographies.csv")
    if sequence_index < len(geography_rows) and geography_rows[sequence_index].get("start_month"):
        return _month_range(geography_rows[sequence_index]["start_month"], target_length)

    entity_rows = _read_csv_rows(dataset_dir / "entities.csv")
    if sequence_index < len(entity_rows):
        row = entity_rows[sequence_index]
        if row.get("start_year"):
            return _year_range(row["start_year"], target_length)
        if row.get("start_month"):
            return _month_range(row["start_month"], target_length)
        if row.get("start_week"):
            return _week_range(row["start_week"], target_length)

    return [f"t{step}" for step in range(1, target_length + 1)]


def build_artifact_labels(dataset_dir: Path, num_sequences: int, prob_dim: int, target_lengths: Sequence[int]) -> Dict[str, List[str] | List[List[str]]]:
    sequence_labels = infer_sequence_labels(dataset_dir, num_sequences)
    token_labels = infer_token_labels(dataset_dir, prob_dim)
    time_labels = [
        infer_time_labels(dataset_dir, sequence_index=idx, target_length=int(target_lengths[idx]))
        for idx in range(num_sequences)
    ]
    return {
        "sequence_labels": sequence_labels,
        "token_labels": token_labels,
        "time_labels": time_labels,
    }
