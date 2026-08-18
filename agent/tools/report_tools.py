"""
Health report generator.

Builds a structured markdown report covering pipeline status, quality,
anomalies, market correlation, and recommended actions.
"""

from datetime import datetime
from langchain_core.tools import tool
from data.query_engine import (
    enrich_roster, filter_roster, aggregate_roster,
    get_stuck_ros, get_failed_ros, cross_table_join,
    compute_record_quality, detect_anomalies, retry_effectiveness,
    resolve_state,
)
from data.loader import get_roster_data, get_market_data


@tool
def generate_health_report(state: str = "", org: str = "") -> str:
    """Generate a Pipeline & Quality Health Report.

    Produces a structured multi-section report: summary stats, stage health,
    source system breakdown, failure analysis, worst orgs, market correlation,
    and recommended next actions.

    Args:
        state: Optional state filter (e.g., 'CA', 'New York'). Blank = national.
        org:   Optional org name filter (partial match).

    Returns:
        Formatted markdown report.
    """
    scope = resolve_state(state) if state else "NATIONAL"
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    roster = enrich_roster(get_roster_data())
    if state:
        resolved = resolve_state(state)
        roster = roster[roster["CNT_STATE"].str.upper() == resolved]
    if org:
        roster = roster[roster["ORG_NM"].str.contains(org, case=False, na=False)]

    n = len(roster)
    if n == 0:
        return f"No data for scope={scope}" + (f", org={org}" if org else "")

    # quick stats
    resolved_n = roster["IS_RESOLVED"].sum()
    res_rate = resolved_n / n * 100
    stuck_n = roster["IS_STUCK"].sum()
    failed_n = roster["IS_FAILED"].sum()
    avg_health = roster["HEALTH_SCORE"].mean()
    avg_red = roster["RED_FLAG_COUNT"].mean()
    retries = roster["IS_RETRY"].sum()
    retry_pct = retries / n * 100

    # stage distribution
    stage_dist = roster["LATEST_STAGE_NM"].value_counts().head(10)

    # per-source-system summary
    src_dist = roster.groupby("SRC_SYS").agg(
        count=("RO_ID", "count"),
        avg_health=("HEALTH_SCORE", "mean"),
        fail_count=("IS_FAILED", "sum"),
    ).sort_values("count", ascending=False)

    # worst orgs
    org_health = roster.groupby("ORG_NM").agg(
        file_count=("RO_ID", "count"),
        avg_health=("HEALTH_SCORE", "mean"),
        red_flags=("RED_FLAG_COUNT", "sum"),
    ).sort_values("avg_health").head(5)

    # failures
    failure_dist = roster[roster["IS_FAILED"]]["FAILURE_STATUS"].value_counts()

    # per-stage red flag %
    health_stages = {}
    for col in ["PRE_PROCESSING_HEALTH", "DART_GEN_HEALTH", "DART_REVIEW_HEALTH",
                 "SPS_LOAD_HEALTH", "DART_UI_VALIDATION_HEALTH"]:
        if col in roster.columns:
            pct = (roster[col].str.strip().str.lower() == "red").mean() * 100
            health_stages[col.replace("_HEALTH", "")] = round(pct, 1)

    # market correlation (if state-level)
    market_section = ""
    if state:
        market = get_market_data()
        mkt = market[market["MARKET"].str.upper() == resolve_state(state)]
        if not mkt.empty:
            avg_scs = mkt["SCS_PERCENT"].mean()
            latest = mkt.sort_values("MONTH").iloc[-1]
            market_section = f"""
## 🌍 Market Performance (CSV2)
| Metric | Value |
|--------|-------|
| Avg SCS% | {avg_scs:.1f}% |
| Latest Month | {latest['MONTH']} |
| Latest SCS% | {latest['SCS_PERCENT']:.1f}% |
| Overall Success | {latest['OVERALL_SCS_CNT']:,.0f} |
| Overall Fail | {latest['OVERALL_FAIL_CNT']:,.0f} |
| Quality Gap | {avg_scs - res_rate:.1f}pp |
"""

    # ---- assemble report ----
    report = f"""
# 📋 Pipeline & Quality Health Report
**Scope:** {scope}{f' | Org: {org}' if org else ''} | **Generated:** {now}

---

## 📊 Summary
| Metric | Value |
|--------|-------|
| Total ROs | {n:,} |
| Resolved | {resolved_n:,} ({res_rate:.1f}%) |
| Failed | {int(failed_n):,} ({failed_n/n*100:.1f}%) |
| Stuck | {int(stuck_n):,} |
| Retried | {int(retries):,} ({retry_pct:.1f}%) |
| Avg Health Score | {avg_health:.1f}/100 |
| Avg Red Flags | {avg_red:.1f}/7 |

---

## ⏱ Stage Distribution
| Stage | Count | % |
|-------|-------|---|
"""
    for stg, cnt in stage_dist.items():
        report += f"| {stg} | {cnt:,} | {cnt/n*100:.1f}% |\n"

    report += "\n---\n\n## 🚨 Stage Health (% Red Flags)\n| Stage | % Red |\n|-------|-------|\n"
    for stg, pct in sorted(health_stages.items(), key=lambda x: x[1], reverse=True):
        icon = "🔴" if pct > 15 else "🟡" if pct > 5 else "🟢"
        report += f"| {icon} {stg} | {pct}% |\n"

    report += "\n---\n\n## 📡 Source Systems\n| Source | Files | Avg Health | Failures |\n|--------|-------|-----------|----------|\n"
    for src, row in src_dist.iterrows():
        report += f"| {src} | {int(row['count']):,} | {row['avg_health']:.1f} | {int(row['fail_count'])} |\n"

    if not failure_dist.empty:
        report += "\n---\n\n## ❌ Failures\n| Reason | Count |\n|--------|-------|\n"
        for reason, cnt in failure_dist.items():
            report += f"| {reason} | {cnt:,} |\n"

    report += "\n---\n\n## 🏢 Worst Orgs (Health)\n| Organization | Files | Health | Red Flags |\n|-------------|-------|--------|----------|\n"
    for org_nm, row in org_health.iterrows():
        report += f"| {org_nm[:35]} | {int(row['file_count'])} | {row['avg_health']:.1f} | {int(row['red_flags'])} |\n"

    if market_section:
        report += market_section

    # recommended actions
    report += "\n---\n\n## 💡 Recommended Actions\n"
    i = 1
    if stuck_n > 0:
        report += f"{i}. **Escalate {int(stuck_n)} stuck RO(s)**\n"; i += 1
    worst = max(health_stages.items(), key=lambda x: x[1]) if health_stages else None
    if worst and worst[1] > 10:
        report += f"{i}. **Investigate {worst[0]}** — {worst[1]}% Red\n"; i += 1
    if retry_pct > 30:
        report += f"{i}. **High retry rate ({retry_pct:.0f}%)** — root-cause first-pass failures\n"; i += 1
    if not failure_dist.empty:
        report += f"{i}. **Address '{failure_dist.index[0]}'** ({failure_dist.iloc[0]:,} cases)\n"; i += 1
    if avg_health < 70:
        report += f"{i}. **Overall health {avg_health:.1f} < 70** — systemic review needed\n"

    report += f"\n---\n*Generated by RosterIQ Agent | {now}*\n"
    return report


ALL_REPORT_TOOLS = [generate_health_report]
