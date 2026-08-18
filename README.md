# RosterIQ — AI Agent for Healthcare Provider Roster Operations

An autonomous AI agent that diagnoses pipeline issues, derives quality metrics on-the-fly, and communicates insights through natural language — powered by 4 memory layers, 29 tools, and Gemini 2.0 Flash.

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/Sathwik1729/Agentic_AI.git
cd Agentic_AI

# 2. Create virtual environment
python -m venv newvenv
source newvenv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add the datasets (see "Data Files" below)
#    Place both CSVs in the project root.

# 5. Configure API keys
cp .env.example .env
# Edit .env and add your keys:
#   GOOGLE_API_KEY=your_gemini_key
#   TAVILY_API_KEY=your_tavily_key (optional, for web search)

# 6. Run
streamlit run app.py
```

### Data Files

The two CSVs are **not included in this repository** and must be supplied locally.
`data/loader.py` reads them from the project root:

| File | Shape | Contents |
|---|---|---|
| `roster_processing_details.csv` | 59,975 x 53 | File-level pipeline records |
| `aggregated_operational_metrics.csv` | 357 x 16 | Market-level rollups |

Both are listed in `.gitignore` so they are never committed.

**Get API keys:**
- **Gemini** (required): [aistudio.google.com](https://aistudio.google.com)
- **Tavily** (optional): [tavily.com](https://tavily.com)

---

## Architecture

```
┌─────────────── Streamlit UI (app.py) ──────────────┐
│  Chat Interface │ Sidebar (Memory Status) │ Charts  │
└────────────────────────┬───────────────────────────┘
                         │
              ┌──────────▼──────────┐
              │   LangGraph Agent   │
              │   (Gemini 2.0 Flash)│
              │   ReAct Loop        │
              └──────────┬──────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
   ┌──────────┐   ┌──────────┐   ┌──────────┐
   │ 29 Tools │   │ 4 Memory │   │ Enriched │
   │          │   │  Layers  │   │   Data   │
   └──────────┘   └──────────┘   └──────────┘
```

### 29 Agent Tools

| Category | Count | Tools |
|----------|-------|-------|
| **Data Query** | 11 | Roster filter, market filter, aggregation, stuck/failed ROs, cross-table join, anomaly detection, record quality, retry analysis (2), domain lookup |
| **Visualization** | 6 | Pipeline heatmap, quality breakdown, duration anomalies, market trends, retry lift, stuck RO tracker |
| **Web Search** | 3 | General search, CMS regulations, provider context |
| **Dynamic Query** | 1 | Agent writes & executes arbitrary Pandas code |
| **Reports** | 1 | Structured health report generator |
| **Procedural Memory** | 3 | List/get/update diagnostic procedures |
| **Episodic Memory** | 4 | Recall interactions, store insights, session summary, recall reflections |

### 4 Memory Layers

| Layer | Storage | Purpose |
|-------|---------|---------|
|  **Semantic** | Hardcoded ontology | Domain knowledge (pipeline stages, status codes, LOBs, CMS context) |
|  **Procedural** | YAML files | Diagnostic workflows (triage, quality audit, market report, retry analysis) |
|  **Episodic** | ChromaDB | Conversation history with semantic search recall |
|  **Reflection** | ChromaDB | Agent-generated meta-insights that make it smarter over time |

---

##  On-the-Fly Metric Derivation

The raw CSV does **not** contain pre-computed record counts. RosterIQ derives all operational signals at query time:

| Derived Metric | Source Signals |
|---|---|
| `HEALTH_SCORE` (0-100) | 7 stage health flags (Green=100, Yellow=50, Red=0) |
| `RED_FLAG_COUNT` (0-7) | Count of "Red" health flags |
| `FILE_OUTCOME` | `FILE_STATUS_CD` + `IS_STUCK` + `IS_FAILED` |
| `IS_RESOLVED` | `FILE_STATUS_CD == 99` |
| `DART_GEN_DEVIATION` | `DART_GEN_DURATION / AVG_DART_GENERATION_DURATION` |
| `IS_RETRY`, `RETRY_COUNT` | `RUN_NO > 1` |

Additionally, the agent can write custom Pandas code via the `run_data_query` tool for any analysis not covered by pre-built tools.

---

##  Project Structure

```
├── app.py                              # Streamlit entry point
├── requirements.txt
├── .env.example                        # Template for .env
├── roster_processing_details.csv       # CSV1: not in repo (gitignored)
├── aggregated_operational_metrics.csv  # CSV2: not in repo (gitignored)
│
├── data/
│   ├── loader.py                       # CSV loading with caching
│   └── query_engine.py                 # On-the-fly metric derivation
│
├── agent/
│   ├── graph.py                        # LangGraph ReAct agent
│   ├── prompts.py                      # System prompt with semantic memory
│   └── tools/
│       ├── data_tools.py               # 11 data query tools
│       ├── viz_tools.py                # 6 Plotly visualization tools
│       ├── web_search.py               # 3 Tavily web search tools
│       ├── dynamic_query.py            # Autonomous Pandas code execution
│       └── report_tools.py             # Health report generator
│
├── memory/
│   ├── semantic.py                     # Domain knowledge ontology
│   ├── procedural.py                   # YAML procedure registry
│   └── episodic.py                     # ChromaDB episodic + reflection memory
│
├── procedures/                         # Stored diagnostic workflows
│   ├── triage_stuck_ros.yaml
│   ├── record_quality_audit.yaml
│   ├── market_health_report.yaml
│   └── retry_effectiveness.yaml
│
└── memory_store/                       # ChromaDB persistent storage (gitignored)
```

---

##  Example Interactions

### Pipeline Triage
> **User:** "Which ROs are stuck and need immediate attention?"  
> **Agent:** Finds stuck ROs → ranks by RED_FLAG_COUNT → suggests escalation priorities

### Record Quality Audit
> **User:** "Show me the quality breakdown for California files"  
> **Agent:** Derives HEALTH_SCORE per file → identifies worst orgs → generates bar chart

### Market Correlation
> **User:** "How does NY's file health correlate with its market success rate?"  
> **Agent:** Cross-joins CSV1 health metrics with CSV2 SCS_PERCENT → shows quality_gap

### Custom Analysis
> **User:** "What's the average retry count per source system in Texas?"  
> **Agent:** Writes Pandas code: `df[df['CNT_STATE']=='TX'].groupby('SRC_SYS')['RETRY_COUNT'].mean()`

### Regulatory Context
> **User:** "Are there any CMS rule changes affecting provider directories?"  
> **Agent:** Searches CMS.gov → returns relevant regulatory updates with citations

---

##  Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Gemini 2.0 Flash |
| Agent Framework | LangGraph (ReAct pattern) |
| UI | Streamlit |
| Memory | ChromaDB (persistent) |
| Data | Pandas |
| Visualizations | Plotly |
| Web Search | Tavily |

---

##  Novel Innovations

1. ** Reflection Memory** — After each diagnostic session, the agent generates meta-insights that are stored separately and retrieved in future sessions, making it genuinely smarter over time.

2. ** True Autonomous Querying** — The agent can write and execute its own Pandas code at runtime via the `run_data_query` tool, enabling it to answer questions we didn't anticipate.

3. ** On-the-Fly Metric Derivation** — Instead of relying on pre-computed columns, the agent derives operational signals from raw data, demonstrating real analytical capability.
