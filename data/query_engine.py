"""
Query engine — derives operational signals on-the-fly from the raw 53-col roster data.

The raw CSV is missing the pre-computed record counts (TOT_REC_CNT, SCS_REC_CNT, etc.).
Instead we derive equivalent signals from FILE_STATUS_CD, the 7 stage health flags,
actual vs average durations, RUN_NO, IS_STUCK, IS_FAILED, etc.
"""

import pandas as pd
import numpy as np
from typing import Optional
from data.loader import get_roster_data, get_market_data


# ---------------------------------------------------------------------------
# State name → abbreviation lookup
# (so users can say "Texas" and we convert to "TX" before filtering)
# ---------------------------------------------------------------------------

STATE_ABBREVS = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
    "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
    "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
    "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY",
    "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK",
    "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI",
    "south carolina": "SC", "south dakota": "SD", "tennessee": "TN", "texas": "TX",
    "utah": "UT", "vermont": "VT", "virginia": "VA", "washington": "WA",
    "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY",
    "district of columbia": "DC", "national": "NATIONAL",
}


def resolve_state(raw):
    """Turn 'Texas' into 'TX', pass through 'TX' unchanged, etc."""
    if not raw:
        return ""
    s = raw.strip()
    if len(s) <= 3:        # already an abbreviation
        return s.upper()
    hit = STATE_ABBREVS.get(s.lower())
    return hit if hit else s.upper()


# ---------------------------------------------------------------------------
# Enrichment — bolts derived columns onto the raw dataframe
# ---------------------------------------------------------------------------

HEALTH_COLS = [
    "PRE_PROCESSING_HEALTH", "MAPPING_APROVAL_HEALTH",
    "ISF_GEN_HEALTH", "DART_GEN_HEALTH", "DART_REVIEW_HEALTH",
    "DART_UI_VALIDATION_HEALTH", "SPS_LOAD_HEALTH",
]

HEALTH_SCORES = {"Green": 100, "Yellow": 50, "Red": 0}

# maps actual duration col -> historical avg col
AVG_BENCHMARKS = {
    "DART_GEN_DURATION": "AVG_DART_GENERATION_DURATION",
    "DART_UI_VALIDATION_DURATION": "AVG_DART_UI_VLDTN_DURATION",
    "SPS_LOAD_DURATION": "AVG_SPS_LOAD_DURATION",
    "ISF_GEN_DURATION": "AVG_ISF_GENERATION_DURATION",
}


def enrich_roster(df):
    """
    Add derived columns to the roster dataframe.

    New cols:
        IS_RESOLVED, RED_FLAG_COUNT, YELLOW_FLAG_COUNT, HEALTH_SCORE,
        MAX_DURATION, *_DEVIATION (duration ratios), IS_RETRY, RETRY_COUNT,
        FILE_OUTCOME (resolved | in_progress | stuck | failed)
    """
    df = df.copy()

    h_cols = [c for c in HEALTH_COLS if c in df.columns]

    # completion status
    df["IS_RESOLVED"] = (df["FILE_STATUS_CD"] == 99.0)

    # health flag counts
    if h_cols:
        df["RED_FLAG_COUNT"] = df[h_cols].apply(
            lambda r: (r.str.strip().str.lower() == "red").sum(), axis=1
        )
        df["YELLOW_FLAG_COUNT"] = df[h_cols].apply(
            lambda r: (r.str.strip().str.lower() == "yellow").sum(), axis=1
        )
        df["HEALTH_SCORE"] = df[h_cols].apply(
            lambda r: r.str.strip().map(HEALTH_SCORES).mean(), axis=1
        ).round(1)
    else:
        df["RED_FLAG_COUNT"] = 0
        df["YELLOW_FLAG_COUNT"] = 0
        df["HEALTH_SCORE"] = 100.0

    # longest stage duration for this file
    dur_cols = [c for c in df.columns if c.endswith("_DURATION") and not c.startswith("AVG_")]
    df["MAX_DURATION"] = df[dur_cols].max(axis=1) if dur_cols else 0.0

    # deviation ratios (actual / benchmark)
    for actual, avg in AVG_BENCHMARKS.items():
        if actual in df.columns and avg in df.columns:
            tag = actual.replace("_DURATION", "_DEVIATION")
            df[tag] = (df[actual] / df[avg].replace(0, np.nan)).round(2)

    # retry info
    df["IS_RETRY"] = df["RUN_NO"] > 1
    df["RETRY_COUNT"] = (df["RUN_NO"] - 1).clip(lower=0)

    # single outcome label
    def _outcome(row):
        if row.get("IS_STUCK", False):
            return "stuck"
        if row.get("IS_FAILED", False):
            return "failed"
        if row.get("IS_RESOLVED", False):
            return "resolved"
        return "in_progress"

    df["FILE_OUTCOME"] = df.apply(_outcome, axis=1)
    return df


# ---------------------------------------------------------------------------
# Filtering helpers
# ---------------------------------------------------------------------------

def filter_roster(
    state=None, org=None, lob=None, src_sys=None, stage=None,
    is_stuck=None, is_failed=None, outcome=None,
    min_health_score=None, max_health_score=None,
    run_no=None, limit=100,
):
    """Flexible filter over the enriched roster data. All params optional."""
    df = enrich_roster(get_roster_data())

    if state:
        df = df[df["CNT_STATE"].str.upper() == resolve_state(state)]
    if org:
        df = df[df["ORG_NM"].str.contains(org, case=False, na=False)]
    if lob:
        df = df[df["LOB"].str.contains(lob, case=False, na=False)]
    if src_sys:
        df = df[df["SRC_SYS"].str.upper() == src_sys.upper()]
    if stage:
        df = df[df["LATEST_STAGE_NM"].str.upper() == stage.upper()]
    if is_stuck is not None:
        df = df[df["IS_STUCK"] == is_stuck]
    if is_failed is not None:
        df = df[df["IS_FAILED"] == is_failed]
    if outcome:
        df = df[df["FILE_OUTCOME"] == outcome.lower()]
    if min_health_score is not None:
        df = df[df["HEALTH_SCORE"] >= min_health_score]
    if max_health_score is not None:
        df = df[df["HEALTH_SCORE"] <= max_health_score]
    if run_no is not None:
        df = df[df["RUN_NO"] == run_no]

    if "FILE_RECEIVED_DT" in df.columns:
        df = df.sort_values("FILE_RECEIVED_DT", ascending=False)

    return df.head(limit)


def filter_market(market=None, month=None, min_scs_percent=None,
                  max_scs_percent=None, is_active=None, limit=100):
    """Filter the market-level aggregated metrics."""
    df = get_market_data()

    if market:
        df = df[df["MARKET"].str.upper() == resolve_state(market)]
    if month:
        df = df[df["MONTH"] == month]
    if min_scs_percent is not None:
        df = df[df["SCS_PERCENT"] >= min_scs_percent]
    if max_scs_percent is not None:
        df = df[df["SCS_PERCENT"] <= max_scs_percent]
    if is_active is not None:
        df = df[df["IS_ACTIVE"] == is_active]

    return df.head(limit)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_roster(group_cols, agg_dict=None, filters=None):
    """Group the enriched roster by given columns with default or custom aggs."""
    if filters:
        df = filter_roster(**filters, limit=999_999)
    else:
        df = enrich_roster(get_roster_data())

    if agg_dict is None:
        agg_dict = {
            "RO_ID": "count",
            "HEALTH_SCORE": "mean",
            "RED_FLAG_COUNT": "mean",
            "IS_RESOLVED": "mean",
            "IS_FAILED": "sum",
            "IS_STUCK": "sum",
            "IS_RETRY": "sum",
            "MAX_DURATION": "mean",
        }

    valid_agg = {k: v for k, v in agg_dict.items() if k in df.columns}
    valid_groups = [c for c in group_cols if c in df.columns]
    if not valid_groups or not valid_agg:
        return pd.DataFrame()

    out = df.groupby(valid_groups, dropna=False).agg(valid_agg).reset_index()
    if "RO_ID" in out.columns:
        out = out.rename(columns={"RO_ID": "RO_COUNT"})
    if "IS_RESOLVED" in out.columns:
        out["RESOLUTION_RATE_PCT"] = (out["IS_RESOLVED"] * 100).round(1)
    return out


# ---------------------------------------------------------------------------
# Stuck / failed shortcuts
# ---------------------------------------------------------------------------

def get_stuck_ros(state=None):
    """All stuck ROs, ranked worst-first by red flags then duration."""
    df = filter_roster(is_stuck=True, state=state, limit=999_999)
    if df.empty:
        return df
    return df.sort_values(["RED_FLAG_COUNT", "MAX_DURATION"], ascending=[False, False])


def get_failed_ros(state=None):
    """All failed ROs with failure reasons and health scores."""
    return filter_roster(is_failed=True, state=state, limit=999_999)


# ---------------------------------------------------------------------------
# Cross-table join (file-level CSV1 ↔ market-level CSV2)
# ---------------------------------------------------------------------------

def cross_table_join(state=None):
    """
    Join file-level health with market-level SCS%, keyed on state.
    Returns one row per state with both sets of metrics + a quality gap.
    """
    roster = enrich_roster(get_roster_data())
    market = get_market_data()

    # roll up roster by state
    r_agg = roster.groupby("CNT_STATE").agg(
        file_count=("RO_ID", "count"),
        avg_health_score=("HEALTH_SCORE", "mean"),
        avg_red_flags=("RED_FLAG_COUNT", "mean"),
        resolution_rate=("IS_RESOLVED", "mean"),
        stuck_count=("IS_STUCK", "sum"),
        failed_count=("IS_FAILED", "sum"),
        retry_count=("IS_RETRY", "sum"),
        avg_max_duration=("MAX_DURATION", "mean"),
    ).reset_index()
    r_agg["resolution_rate_pct"] = (r_agg["resolution_rate"] * 100).round(1)
    r_agg["failure_rate_pct"] = (r_agg["failed_count"] / r_agg["file_count"] * 100).round(1)

    # roll up market
    m_agg = market.groupby("MARKET").agg(
        market_scs_percent=("SCS_PERCENT", "mean"),
        total_market_success=("OVERALL_SCS_CNT", "sum"),
        total_market_fail=("OVERALL_FAIL_CNT", "sum"),
        first_iter_success=("FIRST_ITER_SCS_CNT", "sum"),
        first_iter_fail=("FIRST_ITER_FAIL_CNT", "sum"),
    ).reset_index()

    joined = r_agg.merge(m_agg, left_on="CNT_STATE", right_on="MARKET", how="inner")
    joined["quality_gap"] = (joined["market_scs_percent"] - joined["resolution_rate_pct"]).round(1)

    if state:
        joined = joined[joined["CNT_STATE"].str.upper() == resolve_state(state)]
    return joined


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------

def detect_anomalies(metric="DART_GEN_DURATION", threshold_multiplier=2.0, scope=None):
    """
    Flag rows where `metric` is way outside the norm.

    For duration cols we compare against the AVG_* benchmark.
    For everything else we fall back to z-score.
    """
    df = enrich_roster(get_roster_data())
    if scope:
        df = df[df["CNT_STATE"].str.upper() == resolve_state(scope)]
    if metric not in df.columns:
        return pd.DataFrame()

    df = df.copy()
    vals = df[metric].dropna()
    if vals.empty:
        return pd.DataFrame()

    # duration-based detection
    if metric in AVG_BENCHMARKS and AVG_BENCHMARKS[metric] in df.columns:
        avg_col = AVG_BENCHMARKS[metric]
        df["ANOMALY_RATIO"] = df[metric] / df[avg_col].replace(0, np.nan)
        bad = df[df["ANOMALY_RATIO"] > threshold_multiplier].copy()
        bad["ANOMALY_DETAIL"] = (
            "Actual " + metric + " is "
            + bad["ANOMALY_RATIO"].round(1).astype(str)
            + "x the historical average"
        )
    else:
        # z-score fallback
        mu, sigma = vals.mean(), vals.std()
        if sigma == 0:
            return pd.DataFrame()
        df["Z_SCORE"] = ((df[metric] - mu) / sigma).abs()
        bad = df[df["Z_SCORE"] > threshold_multiplier].copy()
        bad["ANOMALY_DETAIL"] = (
            metric + " = " + bad[metric].round(2).astype(str)
            + " (z=" + bad["Z_SCORE"].round(2).astype(str) + ")"
        )

    return bad.sort_values(metric, ascending=True).head(50)


# ---------------------------------------------------------------------------
# Record quality (derived on-the-fly)
# ---------------------------------------------------------------------------

def compute_record_quality(state=None, org=None, threshold=70.0):
    """
    Per-file quality using derived signals (since raw counts are absent).

    Quality comes from HEALTH_SCORE + FILE_OUTCOME + RED_FLAG_COUNT +
    failure status + retry count.  Files with HEALTH_SCORE < threshold
    get flagged.
    """
    df = filter_roster(state=state, org=org, limit=999_999)
    if df.empty:
        return df

    df = df.copy()
    df["IS_LOW_QUALITY"] = df["HEALTH_SCORE"] < threshold

    # combined severity score — higher = worse
    df["PROBLEM_SCORE"] = (
        df["RED_FLAG_COUNT"] * 15
        + df["IS_FAILED"].astype(int) * 25
        + df["IS_STUCK"].astype(int) * 30
        + df["RETRY_COUNT"].clip(upper=5) * 5
    )

    keep = [
        "RO_ID", "ORG_NM", "CNT_STATE", "LOB", "SRC_SYS",
        "LATEST_STAGE_NM", "FILE_STATUS_CD", "FILE_OUTCOME",
        "HEALTH_SCORE", "RED_FLAG_COUNT", "YELLOW_FLAG_COUNT",
        "MAX_DURATION", "RUN_NO", "IS_RETRY", "IS_FAILED",
        "FAILURE_STATUS", "IS_LOW_QUALITY", "PROBLEM_SCORE",
    ]
    return df[[c for c in keep if c in df.columns]].sort_values("PROBLEM_SCORE", ascending=False)


# ---------------------------------------------------------------------------
# Retry effectiveness
# ---------------------------------------------------------------------------

def retry_effectiveness(state=None):
    """Compare first-iter vs post-retry success rates from market data (CSV2)."""
    market = get_market_data()
    if state:
        market = market[market["MARKET"].str.upper() == resolve_state(state)]

    market = market.copy()
    first_tot = (market["FIRST_ITER_SCS_CNT"] + market["FIRST_ITER_FAIL_CNT"]).replace(0, np.nan)
    all_tot = (market["OVERALL_SCS_CNT"] + market["OVERALL_FAIL_CNT"]).replace(0, np.nan)

    market["FIRST_ITER_SCS_RATE"] = (market["FIRST_ITER_SCS_CNT"] / first_tot * 100).round(2)
    market["POST_RETRY_SCS_RATE"] = (market["OVERALL_SCS_CNT"] / all_tot * 100).round(2)
    market["RETRY_LIFT_PCT"] = (market["POST_RETRY_SCS_RATE"] - market["FIRST_ITER_SCS_RATE"]).round(2)
    market["RECOVERED_BY_RETRY"] = (market["NEXT_ITER_SCS_CNT"] - market["FIRST_ITER_SCS_CNT"]).clip(lower=0)

    cols = [
        "MONTH", "MARKET", "FIRST_ITER_SCS_CNT", "FIRST_ITER_FAIL_CNT",
        "NEXT_ITER_SCS_CNT", "OVERALL_SCS_CNT", "OVERALL_FAIL_CNT",
        "FIRST_ITER_SCS_RATE", "POST_RETRY_SCS_RATE", "RETRY_LIFT_PCT",
        "RECOVERED_BY_RETRY",
    ]
    return market[[c for c in cols if c in market.columns]].sort_values("RETRY_LIFT_PCT", ascending=False)


def roster_retry_analysis(state=None, src_sys=None, org=None):
    """
    File-level retry analysis (CSV1).

    Groups by state + source system + retry flag.  Calculates resolution rates
    AND total pipeline duration so we can quantify 'overhead' properly.
    """
    df = enrich_roster(get_roster_data())

    if state:
        df = df[df["CNT_STATE"].str.upper() == resolve_state(state)]
    if src_sys:
        df = df[df["SRC_SYS"].str.upper() == src_sys.upper()]
    if org:
        df = df[df["ORG_NM"].str.contains(org, case=False, na=False)]

    # total pipeline time for each file
    dur_cols = [c for c in df.columns if c.endswith("_DURATION") and not c.startswith("AVG_")]
    df["TOTAL_PIPELINE_DURATION"] = df[dur_cols].sum(axis=1)

    group_by = ["SRC_SYS", "IS_RETRY"] if src_sys else ["CNT_STATE", "SRC_SYS", "IS_RETRY"]

    summary = df.groupby(group_by).agg(
        file_count=("RO_ID", "count"),
        resolution_rate=("IS_RESOLVED", "mean"),
        failure_rate=("IS_FAILED", "mean"),
        avg_health=("HEALTH_SCORE", "mean"),
        avg_red_flags=("RED_FLAG_COUNT", "mean"),
        total_duration_mins=("TOTAL_PIPELINE_DURATION", "sum"),
        avg_duration_mins=("TOTAL_PIPELINE_DURATION", "mean"),
    ).reset_index()

    summary["resolution_rate_pct"] = (summary["resolution_rate"] * 100).round(1)
    summary["failure_rate_pct"] = (summary["failure_rate"] * 100).round(1)
    summary["total_duration_mins"] = summary["total_duration_mins"].round(0)
    summary["avg_duration_mins"] = summary["avg_duration_mins"].round(1)

    # wasted overhead = duration eaten by retries that still failed
    failed_retries = df[(df["IS_RETRY"]) & (~df["IS_RESOLVED"])]
    if not failed_retries.empty:
        summary.attrs["wasted_retry_duration_mins"] = round(
            failed_retries["TOTAL_PIPELINE_DURATION"].sum(), 0
        )

    return summary.sort_values(group_by)
