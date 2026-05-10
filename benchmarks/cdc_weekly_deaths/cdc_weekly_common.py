from __future__ import annotations

TRACK_NAME = "cdc_weekly_state_cause_share"

DATASET_PAGE_URL = "https://data.cdc.gov/National-Center-for-Health-Statistics/Weekly-Provisional-Counts-of-Deaths-by-State-and-S/muzy-jte6"
DOWNLOAD_CSV_URL = "https://data.cdc.gov/api/views/muzy-jte6/rows.csv?accessType=DOWNLOAD"
METADATA_JSON_URL = "https://data.cdc.gov/api/views/muzy-jte6"

EXCLUDED_JURISDICTIONS = {"United States", "Puerto Rico", "District of Columbia"}
NEW_YORK_COMPONENTS = {"New York", "New York City"}
AGGREGATED_NEW_YORK_NAME = "New York"

SUPPRESSED_MIDPOINT = 5.0
TAIL_DROP_WEEKS = 4
VAL_WEEKS = 52
TEST_WEEKS = 52
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15

SELECTED_CAUSE_COLUMNS = [
    (
        "Diseases of heart",
        "Diseases of heart (I00-I09,I11,I13,I20-I51)",
        "flag_hd",
    ),
    (
        "Malignant neoplasms",
        "Malignant neoplasms (C00-C97)",
        "flag_neopl",
    ),
    (
        "Cerebrovascular diseases",
        "Cerebrovascular diseases (I60-I69)",
        "flag_stroke",
    ),
    (
        "Chronic lower respiratory diseases",
        "Chronic lower respiratory diseases (J40-J47)",
        "flag_clrd",
    ),
    (
        "COVID-19 underlying",
        "COVID-19 (U071, Underlying Cause of Death)",
        "flag_cov19ucod",
    ),
]

OTHER_TOKEN = "OTHER_CAUSES"

