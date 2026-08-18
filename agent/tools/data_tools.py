"""
Data tools – LangChain wrappers around query_engine.

All metrics (HEALTH_SCORE, RED_FLAG_COUNT, FILE_OUTCOME, duration deviations)
are derived on-the-fly from the raw 53 columns.
"""

import json
import pandas as pd
from langchain_core.tools import tool
from data.query_engine import (
    filter_roster,
    filter_market,
    aggregate_roster,
    get_stuck_ros,
    get_failed_ros,
    cross_table_join,
    detect_anomalies,
    compute_record_quality,
    retry_effectiveness,
    roster_retry_analysis,
)
from memory.semantic import lookup_domain_concept


def _df_to_summary(df, max_rows=15):
    """DataFrame → readable markdown string for the agent."""
    if df.empty:
        return "No results found matching the criteria."
    total = len(df)
    table = df.head(max_rows).to_markdown(index=False, floatfmt=".2f")
    header = f"**{total} results found**"
    if total > max_rows:
        header += f" (showing top {max_rows})"
    return f"{header}\n\n{table}"


@tool
def query_roster_data(
    state: str = "",
    org: str = "",
    lob: str = "",
    src_sys: str = "",
    stage: str = "",
    is_stuck: bool = None,
    is_failed: bool = None,
    outcome: str = "",
    min_health_score: float = None,
    max_health_score: float = None,
    run_no: int = None,
    limit: int = 20,
) -> str:
    """Query the roster processing details dataset (CSV1 - file-level pipeline data).
    
    Returns enriched data with derived metrics: HEALTH_SCORE (0-100),
    RED_FLAG_COUNT, FILE_OUTCOME (resolved/in_progress/stuck/failed),
    IS_RESOLVED, IS_RETRY, duration deviations.
    
    Args:
        state: Filter by US state code (e.g., 'CA', 'NY', 'TX')
        org: Filter by organization name (partial match, case-insensitive)
        lob: Filter by Line of Business (e.g., 'Medicare', 'Medicaid')
        src_sys: Filter by source system (e.g., 'AvailityPDM', 'Demographic')
        stage: Filter by current pipeline stage (e.g., 'STOPPED', 'RESOLVED', 'DART_REVIEW')
        is_stuck: If True, only show stuck ROs
        is_failed: If True, only show failed ROs
        outcome: Filter by derived outcome: 'resolved', 'in_progress', 'stuck', 'failed'
        min_health_score: Minimum composite health score (0-100)
        max_health_score: Maximum composite health score (0-100)
        run_no: Filter by processing run number (1 = first pass, >1 = retry)
        limit: Maximum number of results to return (default 20)
    
    Returns:
        Markdown formatted table of matching roster operations.
    """
    df = filter_roster(
        state=state or None,
        org=org or None,
        lob=lob or None,
        src_sys=src_sys or None,
        stage=stage or None,
        is_stuck=is_stuck,
        is_failed=is_failed,
        outcome=outcome or None,
        min_health_score=min_health_score,
        max_health_score=max_health_score,
        run_no=run_no,
        limit=limit,
    )
    
    display_cols = [
        "RO_ID", "ORG_NM", "CNT_STATE", "LOB", "SRC_SYS",
        "LATEST_STAGE_NM", "FILE_OUTCOME", "HEALTH_SCORE",
        "RED_FLAG_COUNT", "RUN_NO", "IS_RETRY",
        "IS_STUCK", "IS_FAILED", "FAILURE_STATUS",
    ]
    existing_cols = [c for c in display_cols if c in df.columns]
    
    return _df_to_summary(df[existing_cols], max_rows=limit)


@tool
def query_market_data(
    market: str = "",
    month: str = "",
    min_scs_percent: float = None,
    max_scs_percent: float = None,
    limit: int = 50,
) -> str:
    """Query the aggregated operational metrics dataset (CSV2 - market-level transactions).
    
    Args:
        market: Filter by market/state (e.g., 'CA', 'NY', 'NATIONAL')
        month: Filter by month (e.g., '01-2026', '12-2025')
        min_scs_percent: Minimum success percentage threshold
        max_scs_percent: Maximum success percentage threshold
        limit: Maximum results to return
    
    Returns:
        Markdown formatted table of market metrics.
    """
    df = filter_market(
        market=market or None,
        month=month or None,
        min_scs_percent=min_scs_percent,
        max_scs_percent=max_scs_percent,
        limit=limit,
    )
    
    display_cols = [
        "MONTH", "MARKET", "FIRST_ITER_SCS_CNT", "FIRST_ITER_FAIL_CNT",
        "OVERALL_SCS_CNT", "OVERALL_FAIL_CNT", "SCS_PERCENT",
    ]
    existing_cols = [c for c in display_cols if c in df.columns]
    
    return _df_to_summary(df[existing_cols], max_rows=limit)


@tool
def aggregate_roster_data(
    group_by: str = "CNT_STATE",
    state: str = "",
) -> str:
    """Aggregate roster data by a grouping column to see summary statistics.
    
    Computes: RO count, avg HEALTH_SCORE, avg RED_FLAG_COUNT,
    resolution rate %, failed count, stuck count, retry count.
    
    Args:
        group_by: Column to group by. Options: 'CNT_STATE', 'SRC_SYS', 'LOB', 
                  'LATEST_STAGE_NM', 'ORG_NM'. Can combine: 'CNT_STATE,SRC_SYS'
        state: Optional state filter.
    
    Returns:
        Markdown table of aggregated statistics.
    """
    group_cols = [g.strip() for g in group_by.split(",")]
    filters = {"state": state} if state else None
    
    df = aggregate_roster(group_cols=group_cols, filters=filters)
    return _df_to_summary(df, max_rows=25)


@tool
def find_stuck_ros(state: str = "") -> str:
    """Find all currently stuck Roster Operations, ranked by severity.
    
    Stuck ROs are sorted by RED_FLAG_COUNT and MAX_DURATION (worst first).
    
    Args:
        state: Optional state filter (e.g., 'CA'). Leave empty for all states.
    
    Returns:
        Table of stuck ROs with severity details.
    """
    df = get_stuck_ros(state=state or None)
    
    if df.empty:
        return "✅ No stuck ROs found" + (f" in {state}" if state else "") + ". All clear!"
    
    display_cols = [
        "RO_ID", "ORG_NM", "CNT_STATE", "LOB", "LATEST_STAGE_NM",
        "HEALTH_SCORE", "RED_FLAG_COUNT", "MAX_DURATION", "FILE_OUTCOME",
        "PRE_PROCESSING_HEALTH", "DART_GEN_HEALTH", "SPS_LOAD_HEALTH",
    ]
    existing_cols = [c for c in display_cols if c in df.columns]
    
    return _df_to_summary(df[existing_cols])


@tool
def find_failed_ros(state: str = "") -> str:
    """Find all failed Roster Operations with failure reasons and health scores.
    
    Args:
        state: Optional state filter. Leave empty for all states.
    
    Returns:
        Table of failed ROs with failure details.
    """
    df = get_failed_ros(state=state or None)
    
    if df.empty:
        return "✅ No failed ROs found" + (f" in {state}" if state else "") + "."
    
    display_cols = [
        "RO_ID", "ORG_NM", "CNT_STATE", "LOB", "SRC_SYS",
        "LATEST_STAGE_NM", "FAILURE_STATUS", "HEALTH_SCORE",
        "RED_FLAG_COUNT", "FILE_OUTCOME",
    ]
    existing_cols = [c for c in display_cols if c in df.columns]
    
    return _df_to_summary(df[existing_cols])


@tool
def correlate_file_and_market_data(state: str = "") -> str:
    """Cross-correlate file-level pipeline health (CSV1) with market-level 
    success percentages (CSV2) for the same state.
    
    Uses derived HEALTH_SCORE and resolution rate instead of raw record counts.
    Shows quality_gap = market_scs_percent - resolution_rate_pct.
    
    Args:
        state: Optional state to focus on. Leave empty for all states.
    
    Returns:
        Joined table showing file-level and market-level metrics side by side.
    """
    df = cross_table_join(state=state or None)
    
    if df.empty:
        return "No cross-table correlation data found" + (f" for {state}" if state else "") + "."
    
    display_cols = [
        "CNT_STATE", "file_count", "avg_health_score", "avg_red_flags",
        "resolution_rate_pct", "failure_rate_pct",
        "market_scs_percent", "quality_gap",
        "stuck_count", "failed_count", "retry_count",
    ]
    existing_cols = [c for c in display_cols if c in df.columns]
    
    return _df_to_summary(df[existing_cols], max_rows=35)


@tool
def find_anomalies(
    metric: str = "DART_GEN_DURATION",
    threshold: float = 2.0,
    state: str = "",
) -> str:
    """Detect anomalies in roster data where a metric significantly 
    deviates from its historical average or statistical norm.
    
    For duration columns, compares actual vs AVG_* benchmark.
    For HEALTH_SCORE, uses z-score detection.
    
    Args:
        metric: Column to analyze. Options: 'DART_GEN_DURATION',
                'ISF_GEN_DURATION', 'SPS_LOAD_DURATION', 'DART_UI_VALIDATION_DURATION',
                'HEALTH_SCORE', 'RED_FLAG_COUNT', 'MAX_DURATION'
        threshold: Multiplier for anomaly detection (default 2.0 = flag values >2x average)
        state: Optional state filter
    
    Returns:
        Table of anomalous ROs with details on why they were flagged.
    """
    df = detect_anomalies(
        metric=metric,
        threshold_multiplier=threshold,
        scope=state or None,
    )
    
    if df.empty:
        return f"No anomalies detected for {metric}" + (f" in {state}" if state else "") + f" at {threshold}x threshold."
    
    display_cols = [
        "RO_ID", "ORG_NM", "CNT_STATE", "SRC_SYS",
        metric, "ANOMALY_DETAIL", "LATEST_STAGE_NM",
    ]
    if "ANOMALY_RATIO" in df.columns:
        display_cols.append("ANOMALY_RATIO")
    if "Z_SCORE" in df.columns:
        display_cols.append("Z_SCORE")
    
    existing_cols = [c for c in display_cols if c in df.columns]
    
    return _df_to_summary(df[existing_cols])


@tool
def analyze_record_quality(
    state: str = "",
    org: str = "",
    threshold: float = 70.0,
) -> str:
    """Analyze per-file record quality using derived signals.
    
    Quality is derived on-the-fly from: HEALTH_SCORE (composite of 7 health flags),
    RED_FLAG_COUNT, FILE_OUTCOME, FAILURE_STATUS, IS_RETRY, and PROBLEM_SCORE.
    
    Args:
        state: Optional state filter
        org: Optional organization name filter (partial match)
        threshold: HEALTH_SCORE below which files are flagged as low quality (default 70)
    
    Returns:
        Table of files with quality breakdown, sorted by worst quality first.
    """
    df = compute_record_quality(
        state=state or None,
        org=org or None,
        threshold=threshold,
    )
    
    if df.empty:
        return "No files found matching criteria."
    
    return _df_to_summary(df, max_rows=20)


@tool
def analyze_retry_effectiveness(state: str = "") -> str:
    """Analyze how effective retries are at recovering failed records.
    
    Uses CSV2 market data to compare first-iteration vs post-retry success rates.
    
    Args:
        state: Optional state/market filter. Leave empty for all markets.
    
    Returns:
        Table showing first-pass vs retry success rates and the lift from retrying.
    """
    df = retry_effectiveness(state=state or None)
    
    if df.empty:
        return "No retry data found" + (f" for {state}" if state else "") + "."
    
    return _df_to_summary(df, max_rows=35)


@tool
def analyze_file_retry_patterns(
    state: str = "",
    src_sys: str = "",
    org: str = "",
) -> str:
    """Analyze retry effectiveness AND overhead at the file level using CSV1.
    
    Compares resolution rates AND total pipeline compute time (overhead)
    between first-pass files (RUN_NO=1) and retried files (RUN_NO>1).
    
    'Overhead' means total pipeline duration consumed by retries,
    especially retries that ultimately failed (wasted compute time).
    
    All filters are optional — omit to analyze the full dataset.
    
    Args:
        state: Optional state filter (e.g., 'CA', 'Texas')
        src_sys: Optional source system filter (e.g., 'AvailityPDM', 'Demographic')
        org: Optional organization name filter (partial match)
    
    Returns:
        Table showing first-pass vs retry resolution rates, failure rates,
        total pipeline duration, and wasted overhead per group.
    """
    df = roster_retry_analysis(
        state=state or None,
        src_sys=src_sys or None,
        org=org or None,
    )
    
    if df.empty:
        return "No file-level retry data found for the specified filters."
    
    result = _df_to_summary(df, max_rows=30)
    
    # Append wasted overhead if available
    wasted = df.attrs.get("wasted_retry_duration_mins")
    if wasted is not None:
        result += f"\n\n**⚠️ Wasted Retry Overhead:** {wasted:,.0f} total pipeline minutes consumed by retries that ultimately failed."
    
    return result


@tool
def lookup_domain_knowledge(term: str) -> str:
    """Look up healthcare roster pipeline domain knowledge.
    
    Use this when you need to explain a concept, status code,
    pipeline stage, health flag, LOB, or source system.
    
    Args:
        term: The concept to look up. Examples: 'STOPPED', 'FILE_OUTCOME',
              'Medicare HMO', 'AvailityPDM', 'Red health flag', 'HEALTH_SCORE'
    
    Returns:
        Detailed explanation of the domain concept.
    """
    return lookup_domain_concept(term)


# all data tools in one list for the agent
ALL_DATA_TOOLS = [
    query_roster_data,
    query_market_data,
    aggregate_roster_data,
    find_stuck_ros,
    find_failed_ros,
    correlate_file_and_market_data,
    find_anomalies,
    analyze_record_quality,
    analyze_retry_effectiveness,
    analyze_file_retry_patterns,
    lookup_domain_knowledge,
]
