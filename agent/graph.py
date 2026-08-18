"""
LangGraph agent setup.

Wires up Gemini 2.5 Flash with all our tools (data, viz, web, memory, etc.)
into a ReAct loop that can reason, call tools, and synthesize answers.
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
try:
    from langchain.agents import create_agent as create_react_agent
    _USES_LANGCHAIN_CREATE_AGENT = True
except Exception:
    from langgraph.prebuilt import create_react_agent
    _USES_LANGCHAIN_CREATE_AGENT = False
from langgraph.checkpoint.memory import MemorySaver
from agent.prompts import get_system_prompt
from agent.tools.data_tools import ALL_DATA_TOOLS
from agent.tools.viz_tools import ALL_VIZ_TOOLS
from agent.tools.web_search import ALL_WEB_TOOLS
from agent.tools.dynamic_query import ALL_DYNAMIC_TOOLS
from agent.tools.report_tools import ALL_REPORT_TOOLS
from memory.procedural import ALL_PROCEDURAL_TOOLS
from memory.episodic import ALL_EPISODIC_TOOLS
from memory.semantic import ALL_SEMANTIC_TOOLS

load_dotenv()


def get_model_candidates() -> list[str]:
    """Return preferred model list for quota-aware fallback."""
    primary = os.getenv("GEMINI_MODEL", "").strip()
    fallback_raw = os.getenv(
        "GEMINI_FALLBACK_MODELS",
        "gemini-2.5-flash,gemini-2.0-flash,gemini-2.0-flash-lite",
    )
    fallback = [m.strip() for m in fallback_raw.split(",") if m.strip()]

    ordered: list[str] = []
    if primary:
        ordered.append(primary)
    for model in fallback:
        if model not in ordered:
            ordered.append(model)
    return ordered or ["gemini-2.5-flash"]


def create_agent(model_name: str | None = None):
    """
    Build the RosterIQ agent.

    ReAct pattern: receive query → decide tools → execute → synthesise answer.
    Short-term memory handled by MemorySaver; long-term by ChromaDB (episodic).
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not found in environment. "
            "Please add it to your .env file. "
            "Get a free key from https://aistudio.google.com"
        )

    resolved_model = model_name or get_model_candidates()[0]

    llm = ChatGoogleGenerativeAI(
        model=resolved_model,
        google_api_key=api_key,
        temperature=0.3,
        max_output_tokens=3072,
        convert_system_message_to_human=False,
    )

    memory = MemorySaver()
    system_prompt = get_system_prompt()

    # pull every tool category into one flat list
    tools = (
        list(ALL_DATA_TOOLS)
        + list(ALL_VIZ_TOOLS)
        + list(ALL_WEB_TOOLS)
        + list(ALL_DYNAMIC_TOOLS)
        + list(ALL_REPORT_TOOLS)
        + list(ALL_PROCEDURAL_TOOLS)
        + list(ALL_EPISODIC_TOOLS)
        + list(ALL_SEMANTIC_TOOLS)
    )

    if _USES_LANGCHAIN_CREATE_AGENT:
        agent = create_react_agent(
            model=llm,
            tools=tools,
            checkpointer=memory,
            system_prompt=system_prompt,
        )
    else:
        agent = create_react_agent(
            model=llm,
            tools=tools,
            checkpointer=memory,
            prompt=system_prompt,
        )
    return agent


def get_agent_config(session_id="default"):
    """Config dict passed to agent.invoke() / agent.stream()."""
    return {
        "configurable": {
            "thread_id": session_id,
        }
    }
