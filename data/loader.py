"""
Loads and caches the two CSV datasets with proper type handling.

We have two data sources:
  1. roster_processing_details.csv   — ~60K rows, file-level pipeline data
  2. aggregated_operational_metrics.csv — ~357 rows, market-level rollups
"""

import pandas as pd
import os
from functools import lru_cache

DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dates that need parsing in the roster file
ROSTER_DATE_COLS = [
    "FILE_RECEIVED_DT", "RA_PLM_RO_FILE_DATA_CREATION",
    "PRE_PROCESSING_START_DT", "PRE_PROCESSING_END_DT",
    "MAPPING_APPRVD_AT",
    "ISF_GEN_START_DT", "ISF_GEN_END_DT",
    "DART_GEN_START_DT", "DART_GEN_END_DT",
    "RELEASED_TO_DART_UI_DT",
    "DART_UI_VALIDATION_START_DT", "DART_UI_VALIDATION_END_DT",
    "SPS_LOAD_START_DT", "SPS_LOAD_END_DT",
    "LATEST_OBJECT_RUN_DT", "CREAT_DT", "LAST_UPDT_DT",
]

# Numeric cols that exist in the raw CSV.
# NOTE: The record-count cols (TOT_REC_CNT, SCS_REC_CNT, etc.) and SCS_PCT
# are NOT present — those get derived on-the-fly in query_engine.py.
ROSTER_NUMERIC_COLS = [
    "PRE_PROCESSING_DURATION", "MAPPING_APROVAL_DURATION",
    "ISF_GEN_DURATION", "DART_GEN_DURATION",
    "DART_REVIEW_DURATION", "DART_UI_VALIDATION_DURATION",
    "SPS_LOAD_DURATION",
    "AVG_DART_GENERATION_DURATION", "AVG_DART_UI_VLDTN_DURATION",
    "AVG_SPS_LOAD_DURATION", "AVG_ISF_GENERATION_DURATION",
    "RUN_NO", "FILE_STATUS_CD",
]

MARKET_DATE_COLS = ["CREAT_DT", "LAST_UPDT_DT"]

MARKET_NUMERIC_COLS = [
    "FIRST_ITER_SCS_CNT", "FIRST_ITER_FAIL_CNT",
    "NEXT_ITER_SCS_CNT", "NEXT_ITER_FAIL_CNT",
    "OVERALL_SCS_CNT", "OVERALL_FAIL_CNT", "SCS_PERCENT",
]


@lru_cache(maxsize=1)
def load_roster_data():
    """Load roster_processing_details.csv — the big one (~60K rows)."""
    path = os.path.join(DATA_DIR, "roster_processing_details.csv")
    df = pd.read_csv(path, low_memory=False)

    # parse dates
    for col in ROSTER_DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # coerce numerics
    for col in ROSTER_NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # booleans
    for col in ["IS_STUCK", "IS_FAILED"]:
        if col in df.columns:
            df[col] = df[col].astype(bool)

    # clean up string cols (strip whitespace)
    for col in ["SRC_SYS", "ORG_NM", "CNT_STATE", "LOB", "LATEST_STAGE_NM", "FAILURE_STATUS"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    health_cols = [c for c in df.columns if c.endswith("_HEALTH")]
    for col in health_cols:
        df[col] = df[col].astype(str).str.strip()

    return df


@lru_cache(maxsize=1)
def load_market_data():
    """Load the market-level aggregated metrics (~357 rows)."""
    path = os.path.join(DATA_DIR, "aggregated_operational_metrics.csv")
    df = pd.read_csv(path, low_memory=False)

    for col in MARKET_DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    for col in MARKET_NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "MARKET" in df.columns:
        df["MARKET"] = df["MARKET"].astype(str).str.strip()
    if "MONTH" in df.columns:
        df["MONTH"] = df["MONTH"].astype(str).str.strip()
    if "IS_ACTIVE" in df.columns:
        df["IS_ACTIVE"] = df["IS_ACTIVE"].astype(bool)

    return df


# convenience wrappers that go through the cache
def get_roster_data():
    return load_roster_data()


def get_market_data():
    return load_market_data()


def get_data_summary():
    """Quick overview of both datasets for the sidebar / agent context."""
    roster = get_roster_data()
    market = get_market_data()

    return {
        "roster": {
            "rows": len(roster),
            "columns": len(roster.columns),
            "states": sorted(roster["CNT_STATE"].dropna().unique().tolist()),
            "source_systems": sorted(roster["SRC_SYS"].dropna().unique().tolist()),
            "stages": sorted(roster["LATEST_STAGE_NM"].dropna().unique().tolist()),
            "stuck_count": int(roster["IS_STUCK"].sum()),
            "failed_count": int(roster["IS_FAILED"].sum()),
            "orgs_count": roster["ORG_NM"].nunique(),
        },
        "market": {
            "rows": len(market),
            "columns": len(market.columns),
            "markets": sorted(market["MARKET"].dropna().unique().tolist()),
            "months": sorted(market["MONTH"].dropna().unique().tolist()),
            "avg_scs_percent": round(market["SCS_PERCENT"].mean(), 2),
        },
    }
