from __future__ import annotations

import os
from typing import Dict

from .base import SequenceTask
from .biotime import BioTIMETask, default_biotime_dir
from .bls_qcew import BLSQCEWTask, default_bls_qcew_dir
from .cdc_weekly_deaths import CDCWeeklyDeathsTask, default_cdc_weekly_deaths_dir
from .ember_monthly import EmberMonthlyTask, default_ember_monthly_dir
from .epa_airdata_aqi import EPAAirDataAQITask, default_epa_airdata_aqi_dir
from .noaa_storm_events import NOAAStormEventsTask, default_noaa_storm_events_dir
from .nyc_tlc_trip import NYCTLCTripTask, default_nyc_tlc_trip_dir
from .owid_dietary import OWIDDietaryTask, default_owid_dietary_dir
from .queue_num_in_sys import QueueNumInSysTask, default_queue_num_in_sys_dir
from .tensor_dir import TensorDirectorySequenceTask

AVAILABLE_TASKS: Dict[str, str] = {
    "biotime": "BioTIME birds annual community-composition benchmark",
    "bls_qcew": "BLS QCEW annual industry-employment share benchmark",
    "cdc_weekly_deaths": "CDC weekly state-level cause-share benchmark",
    "ember_monthly": "Ember monthly electricity generation-mix benchmark",
    "epa_airdata_aqi": "EPA AirData monthly AQI-category share benchmark",
    "noaa_storm_events": "NOAA Storm Events monthly event-type share benchmark",
    "nyc_tlc_trip": "NYC TLC weekly pickup-zone share benchmark",
    "owid_dietary": "OWID annual dietary-composition benchmark",
    "queue_num_in_sys": "Queueing-system occupancy-distribution benchmark",
    "tensor_dir": "Preprocessed probability-vector benchmark stored as train.pt / val.pt tensors",
}


def build_task(name: str) -> SequenceTask:
    normalized = name.strip().lower()
    if normalized in {"tensor_dir", "tensor", "simplex_tensor"}:
        dataset_dir = os.environ.get("DATASET_DIR") or os.environ.get("TENSOR_DATASET_DIR")
        if not dataset_dir:
            raise ValueError(
                "DATASET=tensor_dir requires DATASET_DIR (or TENSOR_DATASET_DIR) to point to train.pt / val.pt files"
            )
        return TensorDirectorySequenceTask(dataset_dir=dataset_dir)

    if normalized in {"biotime", "biotime_birds", "biotime_birds_species_annual"}:
        dataset_dir = os.environ.get("BIOTIME_DIR") or os.environ.get("DATASET_DIR") or default_biotime_dir()
        return BioTIMETask(dataset_dir=dataset_dir)

    if normalized in {"bls_qcew", "qcew", "bls", "qcew_industry_mix"}:
        dataset_dir = os.environ.get("BLS_QCEW_DIR") or os.environ.get("DATASET_DIR") or default_bls_qcew_dir()
        return BLSQCEWTask(dataset_dir=dataset_dir)

    if normalized in {"cdc", "cdc_weekly", "cdc_weekly_deaths", "cdc_mortality"}:
        dataset_dir = (
            os.environ.get("CDC_WEEKLY_DEATHS_DIR")
            or os.environ.get("DATASET_DIR")
            or default_cdc_weekly_deaths_dir()
        )
        return CDCWeeklyDeathsTask(dataset_dir=dataset_dir)

    if normalized in {"ember", "ember_monthly", "ember_electricity", "ember_monthly_generation_mix"}:
        dataset_dir = (
            os.environ.get("EMBER_MONTHLY_DIR")
            or os.environ.get("DATASET_DIR")
            or default_ember_monthly_dir()
        )
        return EmberMonthlyTask(dataset_dir=dataset_dir)

    if normalized in {"epa", "epa_airdata", "epa_airdata_aqi", "airdata_aqi", "aqi"}:
        dataset_dir = (
            os.environ.get("EPA_AIRDATA_AQI_DIR")
            or os.environ.get("DATASET_DIR")
            or default_epa_airdata_aqi_dir()
        )
        return EPAAirDataAQITask(dataset_dir=dataset_dir)

    if normalized in {"noaa_storm_events", "storm_events", "noaa_storms"}:
        dataset_dir = (
            os.environ.get("NOAA_STORM_EVENTS_DIR")
            or os.environ.get("DATASET_DIR")
            or default_noaa_storm_events_dir()
        )
        return NOAAStormEventsTask(dataset_dir=dataset_dir)

    if normalized in {"nyc_tlc", "nyc_tlc_trip", "tlc_trip", "taxi_zone_mix"}:
        dataset_dir = (
            os.environ.get("NYC_TLC_TRIP_DIR")
            or os.environ.get("DATASET_DIR")
            or default_nyc_tlc_trip_dir()
        )
        return NYCTLCTripTask(dataset_dir=dataset_dir)

    if normalized in {"owid", "owid_dietary", "dietary_composition", "owid_diet"}:
        dataset_dir = (
            os.environ.get("OWID_DIETARY_DIR")
            or os.environ.get("DATASET_DIR")
            or default_owid_dietary_dir()
        )
        return OWIDDietaryTask(dataset_dir=dataset_dir)

    if normalized in {"queue", "queue_num_in_sys", "queueing", "queue_occupancy"}:
        dataset_dir = (
            os.environ.get("QUEUE_NUM_IN_SYS_DIR")
            or os.environ.get("DATASET_DIR")
            or default_queue_num_in_sys_dir()
        )
        return QueueNumInSysTask(dataset_dir=dataset_dir)

    available = ", ".join(sorted(AVAILABLE_TASKS))
    raise ValueError(f"unknown DATASET={name!r}. Available tasks: {available}")
