"""
Procedural Memory – Standard diagnostic workflows.

Stores and retrieves YAML files that define multi-step procedures
(e.g., triage_stuck_ros, record_quality_audit).
"""

import os
import yaml
from pathlib import Path
from langchain_core.tools import tool

# procedures live in the repo-level procedures/ directory.
ROOT_PROCEDURES_DIR = Path(__file__).resolve().parent.parent / "procedures"
LEGACY_PROCEDURES_DIR = Path(__file__).resolve().parent / "procedures"
BASE_DIR = ROOT_PROCEDURES_DIR if ROOT_PROCEDURES_DIR.exists() else LEGACY_PROCEDURES_DIR
BASE_DIR.mkdir(exist_ok=True)

def list_all_procedures():
    """Returns list of all available procedure IDs."""
    if not BASE_DIR.exists(): return []
    return [f.stem for f in BASE_DIR.glob("*.yaml")]

def get_procedure(proc_id):
    """Loads a specific procedure YAML."""
    path = BASE_DIR / f"{proc_id}.yaml"
    if not path.exists(): return None
    with open(path, 'r') as f:
        return yaml.safe_load(f)

@tool
def list_procedures() -> str:
    """List all diagnostic procedures RosterIQ can perform."""
    procs = list_all_procedures()
    if not procs: return "No procedures found."
    return f"Available procedures: {', '.join(procs)}"

@tool
def get_diagnostic_procedure(proc_id: str) -> str:
    """Retrieve the steps for a specific procedure ID."""
    proc = get_procedure(proc_id)
    if not proc: return f"Procedure '{proc_id}' not found."
    
    out = f"### 🛠 {proc.get('name', proc_id)}\n"
    out += f"{proc.get('description', '')}\n\n**Steps:**\n"
    for i, step in enumerate(proc.get('steps', [])):
        out += f"{i+1}. {step}\n"
    return out

@tool
def update_procedure(proc_id: str, name: str, description: str, steps: list[str]) -> str:
    """Create or update a diagnostic procedure."""
    data = {"name": name, "description": description, "steps": steps}
    path = BASE_DIR / f"{proc_id}.yaml"
    with open(path, 'w') as f:
        yaml.dump(data, f)
    return f"✅ Procedure '{proc_id}' saved."

ALL_PROCEDURAL_TOOLS = [list_procedures, get_diagnostic_procedure, update_procedure]
