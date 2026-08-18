from datetime import datetime
from pathlib import Path
import time

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
import streamlit as st

load_dotenv()

# Must stay at the top before any Streamlit rendering calls.
st.set_page_config(page_title="RosterIQ", layout="wide", initial_sidebar_state="collapsed")


def _ensure_streamlit_theme_file() -> None:
    """Create/update .streamlit/config.toml with required dark slate theme."""
    root = Path(__file__).resolve().parent
    config_dir = root / ".streamlit"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "config.toml"
    config_file.write_text(
        """[theme]
backgroundColor="#0f172a"
secondaryBackgroundColor="#1e293b"
textColor="#f8fafc"
primaryColor="#6366f1"
font="sans serif"
""",
        encoding="utf-8",
    )


def _inject_styles() -> None:
    st.markdown(
        """
<style>
/* Global shell */
html, body, [class*="css"] {
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
}

.stApp {
    background: #0f172a !important;
    color: #f8fafc !important;
}

header[data-testid="stHeader"] {
    height: 0 !important;
    background: transparent !important;
}

.main > div,
[data-testid="stAppViewContainer"] {
    background: #0f172a !important;
}

.block-container {
    max-width: 100% !important;
    padding-top: 0 !important;
    padding-bottom: 0.5rem !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
}

/* Panel cards */
.surface {
    background: rgba(30, 41, 59, 0.55);
    border: 1px solid #334155;
    border-radius: 12px;
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
}

.canvas-title {
    font-size: 1rem;
    color: #e2e8f0;
    margin: 0 0 0.8rem 0;
    font-weight: 600;
}

/* Chat and status components */
[data-testid="stChatMessage"],
.stChatMessage,
[data-testid="stStatusWidget"] {
    border-radius: 12px !important;
    border: 1px solid #334155 !important;
    background: rgba(15, 23, 42, 0.62) !important;
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
}

[data-testid="stChatInput"] {
    border: 1px solid #334155 !important;
    border-radius: 12px !important;
    background: rgba(15, 23, 42, 0.86) !important;
}

[data-testid="stChatInput"] textarea {
    color: #e2e8f0 !important;
    font-size: 1.05rem !important;
}

[data-testid="stChatInput"] button {
    background: linear-gradient(180deg, #6366f1, #4f46e5) !important;
    border: none !important;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid #334155 !important;
    border-radius: 12px !important;
    background: rgba(30, 41, 59, 0.42) !important;
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
}

/* Metric shell fallback (native st.metric) */
[data-testid="stMetric"] {
    border-radius: 12px !important;
    border: 1px solid #334155 !important;
    background: rgba(15, 23, 42, 0.45) !important;
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
}

/* Dataframe dark inheritance */
[data-testid="stDataFrame"],
[data-testid="stDataFrame"] > div,
[data-testid="stDataFrame"] [role="grid"],
[data-testid="stDataFrameResizable"] {
    background: transparent !important;
    color: #e2e8f0 !important;
    border-color: #334155 !important;
}

/* KPI cards */
.kpi-grid {
    display: flex;
    gap: 0.85rem;
    flex-wrap: wrap;
    margin: 0.2rem 0 0.7rem 0;
}

.kpi-card {
    flex: 1 1 210px;
    min-width: 190px;
    background: linear-gradient(180deg, rgba(15, 23, 42, 0.82), rgba(15, 23, 42, 0.56));
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 0.78rem 0.95rem;
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
}

.kpi-card::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: #6366f1;
}

.kpi-label {
    color: #94a3b8;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.kpi-value {
    color: #f8fafc;
    font-size: 1.28rem;
    margin-top: 0.25rem;
    font-weight: 700;
}

.topbar {
    padding: 0.5rem 0 0.75rem 0;
    margin-bottom: 0.05rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.top-title {
    font-size: 1.75rem;
    font-weight: 800;
    color: #f8fafc;
    margin-bottom: 0.1rem;
    letter-spacing: -0.01em;
}

.top-subtitle {
    color: #94a3b8;
    font-size: 0.92rem;
}

.pill {
    border: 1px solid #334155;
    border-radius: 999px;
    font-size: 0.76rem;
    color: #cbd5e1;
    padding: 0.3rem 0.55rem;
    background: rgba(15, 23, 42, 0.75);
}

.title-stack {
    display: flex;
    flex-direction: column;
}

[data-testid="stStatusWidget"] {
    margin-top: 0.35rem !important;
}

</style>
        """,
        unsafe_allow_html=True,
    )


def _init_state() -> None:
    if "session_start" not in st.session_state:
        st.session_state.session_start = datetime.now().isoformat()
    if "interaction_count" not in st.session_state:
        st.session_state.interaction_count = 0
    if "charts" not in st.session_state:
        st.session_state.charts = []
    if "latest_canvas_charts" not in st.session_state:
        st.session_state.latest_canvas_charts = []
    if "traces" not in st.session_state:
        st.session_state.traces = []
    if "runs" not in st.session_state:
        initial_run_id = "run_1"
        st.session_state.runs = [
            {
                "id": initial_run_id,
                "title": "New chat",
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "messages": [],
            }
        ]
    if "current_run_id" not in st.session_state:
        st.session_state.current_run_id = st.session_state.runs[0]["id"]
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "agent_error" not in st.session_state:
        st.session_state.agent_error = None
    if "model_candidates" not in st.session_state:
        st.session_state.model_candidates = []
    if "current_model_index" not in st.session_state:
        st.session_state.current_model_index = 0


def _get_current_run() -> dict:
    run_id = st.session_state.current_run_id
    for run in st.session_state.runs:
        if run["id"] == run_id:
            return run
    fallback = st.session_state.runs[0]
    st.session_state.current_run_id = fallback["id"]
    return fallback


def _start_new_run() -> None:
    next_id = f"run_{len(st.session_state.runs) + 1}"
    new_run = {
        "id": next_id,
        "title": "New chat",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "messages": [],
    }
    st.session_state.runs.append(new_run)
    st.session_state.current_run_id = next_id


def _init_agent() -> None:
    if st.session_state.agent is None and st.session_state.agent_error is None:
        try:
            from agent.graph import create_agent, get_agent_config, get_model_candidates

            model_candidates = get_model_candidates()
            st.session_state.model_candidates = model_candidates
            st.session_state.current_model_index = 0
            st.session_state.agent = create_agent(model_name=model_candidates[0])
            st.session_state.agent_config = get_agent_config(
                session_id=st.session_state.session_start
            )
        except Exception as exc:
            st.session_state.agent_error = str(exc)


def _to_display_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text" and item.get("text"):
                    parts.append(str(item["text"]))
                elif item.get("content"):
                    parts.append(str(item["content"]))
            else:
                parts.append(str(item))
        merged = "\n\n".join([p for p in parts if p and p.strip()])
        return merged if merged else str(content)
    return str(content)


def _truncate(text: str, limit: int = 180) -> str:
    value = str(text)
    if len(value) <= limit:
        return value
    return value[:limit] + "…"


def _format_exception(exc: Exception) -> str:
    message = str(exc).strip()
    if not message:
        message = repr(exc)
    kind = exc.__class__.__name__
    text = f"{kind}: {message}"
    lower = text.lower()
    if "429" in lower or "quota" in lower or "rate" in lower or "resource_exhausted" in lower:
        text += "\n\nTip: quota/rate limit likely hit. Retry with a shorter prompt or fewer requested outputs."
    return text


def _is_quota_error(exc: Exception) -> bool:
    lower = f"{exc}".lower()
    return (
        "429" in lower
        or "resource_exhausted" in lower
        or "quota" in lower
        or "rate limit" in lower
    )


def _response_generator(text: str):
    """Yield response tokens for live typewriter streaming."""
    for token in str(text).split(" "):
        yield token + " "
        time.sleep(0.012)


def _build_memory_context(prompt: str) -> tuple[str, list[str]]:
    """Collect relevant episodic, procedural, and semantic context for a prompt."""
    prompt_lower = prompt.lower()
    sections: list[str] = []
    notes: list[str] = []

    try:
        from memory.episodic import get_episodic_memory

        hits = get_episodic_memory().recall(prompt, n=2)
        if hits:
            notes.append(f"Loaded {len(hits)} episodic memory hit(s).")
            lines = []
            for hit in hits[:2]:
                meta = hit.get("meta", {})
                stamp = meta.get("timestamp", "unknown time")
                lines.append(f"- {stamp}: {_truncate(hit.get('content', ''), 220)}")
            sections.append("## Episodic Memory\n" + "\n".join(lines))
        else:
            notes.append("No relevant episodic memory hits found.")
    except Exception as exc:
        notes.append(f"Episodic memory unavailable: {_truncate(exc, 120)}")

    try:
        from memory.procedural import get_procedure, list_all_procedures

        available = list_all_procedures()
        matched: list[str] = []
        for proc_id in available:
            if proc_id.lower() in prompt_lower or proc_id.replace("_", " ").lower() in prompt_lower:
                matched.append(proc_id)

        heuristic_map = {
            "stuck": "triage_stuck_ros",
            "quality": "record_quality_audit",
            "report": "market_health_report",
            "retry": "retry_effectiveness",
        }
        for keyword, proc_id in heuristic_map.items():
            if keyword in prompt_lower and proc_id in available and proc_id not in matched:
                matched.append(proc_id)

        matched = matched[:2]
        if matched:
            notes.append(f"Matched procedural memory: {', '.join(matched)}.")
            proc_blocks = []
            for proc_id in matched:
                proc = get_procedure(proc_id) or {}
                steps = proc.get("steps", [])[:4]
                rendered_steps = "\n".join(f"- {step}" for step in steps)
                proc_blocks.append(
                    f"### {proc.get('name', proc_id)}\n{proc.get('description', '')}\n{rendered_steps}"
                )
            sections.append("## Procedural Memory\n" + "\n\n".join(proc_blocks))
    except Exception as exc:
        notes.append(f"Procedural memory unavailable: {_truncate(exc, 120)}")

    try:
        from memory.semantic import lookup_domain_concept

        semantic_terms = [
            "health", "overhead", "retry", "status", "dart_review",
            "dart_gen", "sps_load", "pre_processing",
        ]
        semantic_hits = []
        for term in semantic_terms:
            if term.replace("_", " ") in prompt_lower or term in prompt_lower:
                semantic_hits.append(f"- {lookup_domain_concept(term)}")
        if semantic_hits:
            notes.append(f"Injected {len(semantic_hits)} semantic hint(s).")
            sections.append("## Semantic Memory\n" + "\n".join(semantic_hits[:3]))
    except Exception as exc:
        notes.append(f"Semantic memory unavailable: {_truncate(exc, 120)}")

    return "\n\n".join(sections), notes


def _build_agent_input(prompt: str, memory_context: str) -> str:
    """Compose the final LLM input with auto-loaded memory context."""
    if not memory_context:
        return prompt

    return (
        f"{prompt}\n\n"
        "Use the auto-loaded memory below if it is relevant. Prioritize direct tool results when they conflict with recalled context.\n\n"
        f"{memory_context}"
    )


def _store_reflection(prompt: str, response: str, tools_used: list[str]) -> None:
    """Persist a compact reflection after a substantive interaction."""
    if not response or response.startswith("❌"):
        return

    compact = " ".join(response.split())
    if len(compact) < 80:
        return

    tags = list(dict.fromkeys([t for t in tools_used[:3] if t]))
    reflection = (
        f"Prompt: {_truncate(prompt, 140)} | "
        f"Outcome: {_truncate(compact, 220)}"
    )

    try:
        from memory.episodic import get_episodic_memory

        get_episodic_memory().store_insight(reflection, tags=tags)
    except Exception:
        pass


def _style_plotly_figure(fig):
    try:
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#94a3b8",
        )
        fig.update_xaxes(gridcolor="rgba(148,163,184,0.14)", zerolinecolor="rgba(148,163,184,0.14)")
        fig.update_yaxes(gridcolor="rgba(148,163,184,0.14)", zerolinecolor="rgba(148,163,184,0.14)")
    except Exception:
        pass
    return fig


def _render_kpi_cards(summary: dict) -> None:
    cards = [
        ("Roster Files", f"{summary['roster']['rows']:,}"),
        ("Stuck ROs", f"{summary['roster']['stuck_count']:,}"),
        ("Failed ROs", f"{summary['roster']['failed_count']:,}"),
        ("Market Records", f"{summary['market']['rows']:,}"),
        ("Avg Market SCS%", f"{summary['market']['avg_scs_percent']}%"),
    ]
    html = ['<div class="kpi-grid">']
    for label, value in cards:
        html.append(
            f'<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div></div>'
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def _process_prompt(prompt: str) -> None:
    st.session_state.interaction_count += 1
    current_run = _get_current_run()
    user_msg_id = f"u_{st.session_state.interaction_count}"
    current_interaction_id = f"ai_{st.session_state.interaction_count}"
    response = ""
    tools_used: list[str] = []
    interaction_charts = []
    memory_notes: list[str] = []

    # Render + persist user message first so thinking appears in-chat immediately.
    with st.chat_message("user"):
        st.markdown(prompt)
    current_run["messages"].append(
        {"role": "user", "content": prompt, "msg_id": user_msg_id}
    )
    if current_run.get("title") == "New chat":
        current_run["title"] = _truncate(prompt, 38)

    if st.session_state.agent is None:
        response = (
            "⚠️ Agent is not connected. "
            "Please add your `GOOGLE_API_KEY` to `.env` and restart the app."
        )
    else:
        st.session_state.charts = []
        consume_generated_charts = None
        memory_context = ""
        try:
            from agent.tools.viz_tools import clear_generated_charts, consume_generated_charts

            clear_generated_charts()
        except Exception:
            pass

        with st.chat_message("assistant"):
            with st.status("Supervisor routing request to specialized agents...", expanded=True) as status:
                status.write("Analyzing intent and selecting execution path.")
                try:
                    status.write("Loading episodic, procedural, and semantic memory...")
                    memory_context, memory_notes = _build_memory_context(prompt)
                    for note in memory_notes:
                        status.write(note)

                    with st.spinner("Thinking..."):
                        result = st.session_state.agent.invoke(
                            {"messages": [HumanMessage(content=_build_agent_input(prompt, memory_context))]},
                            config=st.session_state.agent_config,
                        )

                    for m in result.get("messages", []):
                        if isinstance(m, AIMessage) and m.tool_calls:
                            for tc in m.tool_calls:
                                tool_name = str(tc.get("name", "tool"))
                                tools_used.append(tool_name)
                                status.write(f"Supervisor routing to {tool_name} agent...")
                        elif isinstance(m, ToolMessage):
                            status.write(f"Received tool output: {_truncate(m.content)}")

                    final_messages = [
                        m
                        for m in result.get("messages", [])
                        if isinstance(m, AIMessage) and m.content and not m.tool_calls
                    ]
                    if final_messages:
                        response = _to_display_text(final_messages[-1].content)
                    else:
                        response = "I processed your request but could not produce a final response. Please try rephrasing."

                    interaction_charts = st.session_state.charts.copy()
                    if consume_generated_charts is not None:
                        fallback_charts = consume_generated_charts()
                        if fallback_charts:
                            for chart in fallback_charts:
                                if not any(
                                    id(existing.get("fig")) == id(chart.get("fig"))
                                    for existing in interaction_charts
                                ):
                                    interaction_charts.append(chart)

                    status.update(label="Execution complete", state="complete", expanded=False)
                except Exception as exc:
                    if _is_quota_error(exc):
                        try:
                            from agent.graph import create_agent

                            models = st.session_state.get("model_candidates", [])
                            current_idx = int(st.session_state.get("current_model_index", 0))
                            if current_idx + 1 < len(models):
                                next_idx = current_idx + 1
                                next_model = models[next_idx]
                                st.session_state.agent = create_agent(model_name=next_model)
                                st.session_state.current_model_index = next_idx
                                response = (
                                    f"❌ Error: {_format_exception(exc)}\n\n"
                                    f"I switched to fallback model `{next_model}`. "
                                    "Send the same prompt again now."
                                )
                            else:
                                response = (
                                    f"❌ Error: {_format_exception(exc)}\n\n"
                                    "All configured models are currently quota-limited. "
                                    "Wait briefly and retry."
                                )
                        except Exception as fallback_exc:
                            response = (
                                f"❌ Error: {_format_exception(exc)}\n\n"
                                f"Fallback switch failed: {_format_exception(fallback_exc)}"
                            )
                    else:
                        response = f"❌ Error: {_format_exception(exc)}\n\nPlease try again or rephrase your question."
                    status.update(label="Execution failed", state="error", expanded=True)

            if response.startswith("❌ Error:"):
                st.markdown(response)
            else:
                streamed_response = st.write_stream(_response_generator(response))
                if isinstance(streamed_response, str) and streamed_response.strip():
                    response = streamed_response

                if interaction_charts:
                    for idx, chart_info in enumerate(interaction_charts):
                        fig = chart_info.get("fig")
                        if fig is None:
                            continue
                        fig = _style_plotly_figure(fig)
                        title = chart_info.get("title")
                        if title:
                            st.caption(title)
                        st.plotly_chart(
                            fig,
                            use_container_width=True,
                            key=f"live_chart_{current_interaction_id}_{idx}",
                        )

        st.session_state.charts = []

    st.session_state.latest_canvas_charts = interaction_charts
    current_run["messages"].append(
        {
            "role": "assistant",
            "content": response,
            "msg_id": current_interaction_id,
            "charts": interaction_charts,
        }
    )

    trace = {
        "prompt": _truncate(prompt, 160),
        "tools": tools_used[:5],
        "memory": memory_notes[:5],
        "charts": [c.get("title") or "Chart" for c in interaction_charts][:4],
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    st.session_state.traces.append(trace)
    st.session_state.traces = st.session_state.traces[-8:]

    try:
        from memory.episodic import get_episodic_memory

        mem = get_episodic_memory()
        mem.log_interaction(
            user_query=prompt,
            agent_response=response,
            tools_used=tools_used,
            session_id=st.session_state.session_start,
        )
        _store_reflection(prompt, response, tools_used)
    except Exception:
        pass


_ensure_streamlit_theme_file()
_inject_styles()
_init_state()
_init_agent()

st.markdown(
    """
<div class="topbar">
    <div class="title-stack">
        <div class="top-title">RosterIQ</div>
        <div class="top-subtitle">Multi-agent provider roster intelligence for pipeline health, record quality, and market diagnostics.</div>
    </div>
    <div class="pill">Enterprise Dark UI</div>
</div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Run History")
    if st.button("＋ New run", use_container_width=True):
        _start_new_run()

    st.markdown("---")
    for run in reversed(st.session_state.runs):
        is_active = run["id"] == st.session_state.current_run_id
        label = f"● {run['title']}" if is_active else run["title"]
        if st.button(label, key=f"run_{run['id']}", use_container_width=True):
            st.session_state.current_run_id = run["id"]

current_run = _get_current_run()

st.markdown('<div class="canvas-title">Agent Chat</div>', unsafe_allow_html=True)

prompt = None
chat_box = st.container(height=720, border=True)
with chat_box:
    if st.session_state.agent_error:
        st.warning(
            f"⚠️ Agent not connected: {st.session_state.agent_error}"
        )

    if not current_run["messages"]:
        st.caption("Start a conversation to see assistant responses and status traces.")

    for message in current_run["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(str(message.get("content", "")))
            charts = message.get("charts", [])
            if charts:
                for idx, chart_info in enumerate(charts):
                    fig = chart_info.get("fig")
                    if fig is None:
                        continue
                    fig = _style_plotly_figure(fig)
                    title = chart_info.get("title")
                    if title:
                        st.caption(title)
                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                        key=f"hist_chart_{message.get('msg_id', 'm')}_{idx}",
                    )

    live_area = st.empty()

prompt = st.chat_input(
    "Ask about roster operations, pipeline health, or market trends...",
    key="main_chat_input",
)

if prompt:
    with live_area.container():
        _process_prompt(prompt)
