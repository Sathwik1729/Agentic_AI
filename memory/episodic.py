"""
Episodic Memory – Remembering past conversations.

Uses ChromaDB to store interactions and agent reflections so the agent
can recall previous findings and avoid repeating itself.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import tool


class EpisodicMemory:
    def __init__(self):
        persist_dir = Path(__file__).resolve().parent.parent / "memory_store"
        persist_dir.mkdir(parents=True, exist_ok=True)

        # Local HF embedding model to avoid external embedding API availability issues.
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.db = Chroma(
            # new collection avoids vector-dimension mismatch with older Gemini embeddings
            collection_name="roster_interactions_hf",
            embedding_function=self.embeddings,
            persist_directory=str(persist_dir),
        )

    def log_interaction(self, user_query, agent_response, tools_used, session_id):
        """Store a Q&A pair with metadata."""
        text = f"USER: {user_query}\nAGENT: {agent_response}"
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "tools": json.dumps(tools_used),
            "session_id": session_id,
            "type": "interaction"
        }
        self.db.add_texts([text], [metadata])

    def store_insight(self, insight, tags=None):
        """Store a high-level finding or reflection."""
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "tags": json.dumps(tags or []),
            "type": "insight"
        }
        self.db.add_texts([insight], [metadata])

    def recall(self, query, n=3):
        """Find the most relevant past interactions/insights."""
        results = self.db.similarity_search(query, k=n)
        return [{"content": r.page_content, "meta": r.metadata} for r in results]

# Global singleton
_MEM = None
def get_episodic_memory():
    global _MEM
    if _MEM is None: _MEM = EpisodicMemory()
    return _MEM

@tool
def search_past_findings(query: str) -> str:
    """Search for previous findings, results, or user feedback.
    
    Use this to see if we've already diagnosed a problem or analyzed a state.
    """
    mem = get_episodic_memory()
    hits = mem.recall(query)
    if not hits: return "No relevant past findings found."
    
    out = "### Relevant Past Findings\n\n"
    for h in hits:
        out += f"- {h['content'][:500]}...\n"
    return out

@tool
def store_agent_reflection(reflection: str, tags: Optional[list[str]] = None) -> str:
    """Store an insight or reflection for future sessions."""
    mem = get_episodic_memory()
    mem.store_insight(reflection, tags)
    return "✅ Insight stored in episodic memory."

ALL_EPISODIC_TOOLS = [search_past_findings, store_agent_reflection]
