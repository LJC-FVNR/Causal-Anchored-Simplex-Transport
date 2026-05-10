#!/usr/bin/env bash
set -euo pipefail

DATASETS=${DATASETS:-"biotime ember_monthly owid_dietary cdc_weekly_deaths bls_qcew epa_airdata_aqi noaa_storm_events nyc_tlc_trip"}

for dataset in ${DATASETS}; do
  python "benchmarks/${dataset}/download.py"
  python "benchmarks/${dataset}/prepare.py"
done
