"""
Semantic Memory – Domain knowledge for the RosterIQ agent.

This file holds the "ground truth" for status codes, pipeline stages,
health flags, and other healthcare-specific concepts.
"""

from langchain_core.tools import tool

# -- Status Code Map --
# Derived outcomes: resolved, in_progress, stuck, failed
# Raw data has codes like 99, 65, 9, etc.
FILE_STATUS_MAP = {
    99: "RESOLVED",
    65: "SPS_LOAD",  # currently loading into backend
    9: "STOPPED",   # blocked/parked
    1: "PENDING",
    0: "SUBMITTED",
}

# -- Health Flags --
# Derived from actual vs avg duration benchmarks.
HEALTH_FLAGS = {
    "Green": "Processing within normal time ranges (avg).",
    "Yellow": "Mild delay (1-2x avg duration). Review suggested.",
    "Red": "Significant delay (>2x avg) or critical failure. Action needed.",
}

# -- Pipeline Stages --
PIPELINE_STAGES = {
    "PRE_PROCESSING": "Initial file validation and formatting.",
    "MAPPING_APPROVAL": "Validating file columns against system maps.",
    "ISF_GEN": "Internal Standard Format generation.",
    "DART_GEN": "Business rule validation and record processing.",
    "DART_REVIEW": "Manual or automated review of flagged records.",
    "DART_UI_VALIDATION": "Final system checks before load.",
    "SPS_LOAD": "Pushing data to the production provider system.",
}

# -- Source Systems --
SOURCE_SYSTEMS = {
    "AvailityPDM": "Provider Data Management portal (most common).",
    "Demographic": "Direct demographic feed updates.",
    "Roster": "Bulk roster file uploads (Excels/CSVs).",
}

# -- Knowledge Lookup --
def lookup_domain_concept(term: str) -> str:
    """Answers 'what is X?' queries using hardcoded domain context."""
    t = term.strip().lower()
    
    # helper for partial match
    def find(dct):
        for k, v in dct.items():
            if str(k).lower() in t or t in str(k).lower():
                return f"**{k}**: {v}"
        return None

    res = find(PIPELINE_STAGES) or find(FILE_STATUS_MAP) or find(SOURCE_SYSTEMS)
    if res: return res
    
    if "health" in t:
        return "**Health Scoring**: We derive health on-the-fly. 100=All Green, 0=All Red. " \
               "Red flags are triggered if a stage duration > 2x the benchmark."
    
    if "overhead" in t:
        return "**Overhead**: The total pipeline compute time (in minutes) wasted on " \
               "failed processing runs. We calculate this by summing all duration columns."
               
    return "I don't have a specific definition for that, but it's likely a provider-specific field."


@tool
def lookup_domain_knowledge(term: str) -> str:
    """Look up healthcare roster pipeline concepts, status codes, stages, and operational definitions."""
    return lookup_domain_concept(term)

def get_system_prompt_knowledge():
    """Returns a block of text explaining these concepts for the LLM prompt."""
    return f"""
<DOMAIN_KNOWLEDGE>
PIPELINE STAGES: {list(PIPELINE_STAGES.keys())}
STATUS CODES: {FILE_STATUS_MAP}
SOURCE SYSTEMS: {list(SOURCE_SYSTEMS.keys())}
HEALTH DEFINITION: 7 flags (Green/Yellow/Red). Health Score (0-100) is a composite.
OVERHEAD DEFINITION: Sum of all duration columns. Represents wasted compute time.
</DOMAIN_KNOWLEDGE>
"""


ALL_SEMANTIC_TOOLS = [lookup_domain_knowledge]
