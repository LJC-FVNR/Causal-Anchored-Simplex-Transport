from __future__ import annotations

from dataclasses import dataclass
import sys
from typing import Dict, Optional


@dataclass(frozen=True)
class DatasetDefault:
    canonical_name: str
    block_size: int
    time_unit: str
    rationale: str


DATASET_DEFAULTS: Dict[str, DatasetDefault] = {
    "biotime": DatasetDefault(
        canonical_name="biotime",
        block_size=8,
        time_unit="year",
        rationale="short-to-medium annual ecological panels; preserve target windows across the shortest retained sites",
    ),
    "ember_monthly": DatasetDefault(
        canonical_name="ember_monthly",
        block_size=24,
        time_unit="month",
        rationale="monthly calendar panel; use two annual cycles",
    ),
    "owid_dietary": DatasetDefault(
        canonical_name="owid_dietary",
        block_size=24,
        time_unit="year",
        rationale="long annual dietary panel; use a 24-year calendar history",
    ),
    "cdc_weekly_deaths": DatasetDefault(
        canonical_name="cdc_weekly_deaths",
        block_size=52,
        time_unit="week",
        rationale="weekly mortality panel; use a full annual seasonal cycle",
    ),
    "bls_qcew": DatasetDefault(
        canonical_name="bls_qcew",
        block_size=8,
        time_unit="year",
        rationale="short annual public panel; preserve target windows across all state sequences",
    ),
    "epa_airdata_aqi": DatasetDefault(
        canonical_name="epa_airdata_aqi",
        block_size=24,
        time_unit="month",
        rationale="monthly calendar panel; use two annual cycles",
    ),
    "noaa_storm_events": DatasetDefault(
        canonical_name="noaa_storm_events",
        block_size=24,
        time_unit="month",
        rationale="monthly calendar panel; use two annual cycles",
    ),
    "nyc_tlc_trip": DatasetDefault(
        canonical_name="nyc_tlc_trip",
        block_size=26,
        time_unit="week",
        rationale="weekly calendar panel; use a half-year history while leaving held-out weeks",
    ),
    "queue_num_in_sys": DatasetDefault(
        canonical_name="queue_num_in_sys",
        block_size=128,
        time_unit="simulation step",
        rationale="long synthetic queueing trajectories; use the established queue context length",
    ),
}


ALIASES: Dict[str, str] = {
    "biotime_birds": "biotime",
    "biotime_birds_species_annual": "biotime",
    "ember": "ember_monthly",
    "ember_electricity": "ember_monthly",
    "ember_monthly_generation_mix": "ember_monthly",
    "owid": "owid_dietary",
    "dietary_composition": "owid_dietary",
    "owid_diet": "owid_dietary",
    "cdc": "cdc_weekly_deaths",
    "cdc_weekly": "cdc_weekly_deaths",
    "cdc_mortality": "cdc_weekly_deaths",
    "qcew": "bls_qcew",
    "bls": "bls_qcew",
    "qcew_industry_mix": "bls_qcew",
    "epa": "epa_airdata_aqi",
    "epa_airdata": "epa_airdata_aqi",
    "airdata_aqi": "epa_airdata_aqi",
    "aqi": "epa_airdata_aqi",
    "storm_events": "noaa_storm_events",
    "noaa_storms": "noaa_storm_events",
    "nyc_tlc": "nyc_tlc_trip",
    "tlc_trip": "nyc_tlc_trip",
    "taxi_zone_mix": "nyc_tlc_trip",
    "queue": "queue_num_in_sys",
    "queueing": "queue_num_in_sys",
    "queue_occupancy": "queue_num_in_sys",
}


def canonical_dataset_name(name: str) -> str:
    normalized = name.strip().lower()
    return ALIASES.get(normalized, normalized)


def dataset_default(name: str) -> Optional[DatasetDefault]:
    return DATASET_DEFAULTS.get(canonical_dataset_name(name))


def default_block_size(name: str, fallback: int = 1024) -> int:
    default = dataset_default(name)
    return fallback if default is None else default.block_size


def main(argv: list[str]) -> int:
    if len(argv) == 3 and argv[1] == "block-size":
        print(default_block_size(argv[2]))
        return 0
    if len(argv) == 2 and argv[1] == "table":
        for name, default in sorted(DATASET_DEFAULTS.items()):
            print(f"{name}\t{default.block_size}\t{default.time_unit}\t{default.rationale}")
        return 0
    print("usage: python -m training.dataset_defaults block-size DATASET | table", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
