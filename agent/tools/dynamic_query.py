"""
Sandboxed dynamic query tool.

Lets the agent write & execute arbitrary Pandas code against the
enriched roster and market DataFrames.  This is the "true autonomy"
escape hatch — if none of our pre-built tools cover a user's question,
the agent can just write the code itself.
"""

import io, sys, traceback
import pandas as pd
import numpy as np
from langchain_core.tools import tool
from data.loader import get_roster_data, get_market_data
from data.query_engine import enrich_roster

# things we never want an LLM to run
_BLOCKED = [
    "import os", "import sys", "subprocess", "exec(", "eval(",
    "open(", "__import__", "shutil", "rmdir", "unlink",
    "system(", "popen(", "remove(", "write(",
]


@tool
def run_data_query(python_code: str) -> str:
    """Execute arbitrary Pandas code against the roster and market datasets.

    Use this when the pre-built query tools don't cover your needs.
    You can write any valid Pandas/Python code to derive custom metrics,
    run complex aggregations, or do free-form analysis.

    Available variables:
        - df       : enriched roster DataFrame (~60K × 65 cols), includes
                      HEALTH_SCORE, RED_FLAG_COUNT, FILE_OUTCOME, IS_RETRY, etc.
        - market_df: market metrics DataFrame (~357 × 16 cols)
        - pd, np   : pandas and numpy

    Your code MUST assign the final output to a variable called `result`.

    Args:
        python_code: Valid Python/Pandas code ending with `result = ...`

    Returns:
        The result formatted as markdown.
    """
    code_lower = python_code.lower()
    for kw in _BLOCKED:
        if kw in code_lower:
            return f"⚠️ Blocked: `{kw}` is not allowed for safety. Stick to Pandas/NumPy."

    if "result" not in python_code:
        return "⚠️ Your code must set `result = ...` so I can display the output."

    try:
        roster_df = enrich_roster(get_roster_data())
        market_df = get_market_data()
    except Exception as e:
        return f"⚠️ Data loading failed: {e}"

    env = {"df": roster_df, "market_df": market_df, "pd": pd, "np": np}

    old_stdout = sys.stdout
    sys.stdout = captured = io.StringIO()

    try:
        exec(python_code, env)
        result = env.get("result")
        printed = captured.getvalue()

        if result is None:
            return f"```\n{printed}\n```" if printed else "⚠️ No result. Use `result = ...`."

        if isinstance(result, pd.DataFrame):
            if len(result) > 30:
                out = f"**{len(result)} rows** (first 30)\n\n"
                out += result.head(30).to_markdown(index=True, floatfmt=".2f")
            else:
                out = result.to_markdown(index=True, floatfmt=".2f")
        elif isinstance(result, pd.Series):
            out = result.to_markdown(floatfmt=".2f")
        else:
            out = str(result)

        if printed:
            out = f"```\n{printed}\n```\n\n{out}"
        return out

    except Exception:
        return f"❌ Error:\n```\n{traceback.format_exc().splitlines()[-1]}\n```\nCheck column names / types."
    finally:
        sys.stdout = old_stdout


ALL_DYNAMIC_TOOLS = [run_data_query]
