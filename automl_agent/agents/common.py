"""Helpers shared by all agents."""
from __future__ import annotations

from pathlib import Path

from automl_agent import config, db
from automl_agent.rag import agentic_retrieve, format_context


def run_dir(state) -> Path:
    d = config.RUNS_DIR / state["run_id"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def n_classes(state) -> int | None:
    p = state["profile"]
    return p.get("n_classes") if p["task_type"] == "classification" else None


def trace(state, agent: str, message: str, event: str = "message", detail=None) -> list[dict]:
    """Log to SQLite audit table and return a trace entry for the graph state."""
    db.log(state["run_id"], agent, event, detail if detail is not None else message)
    return [{"agent": agent, "event": event, "message": message, "ts": db.now()}]


def retrieve(queries: list[str]):
    """Agentic RAG call -> (context string for LLM, citations list, retrieval log)."""
    hits, rlog = agentic_retrieve(queries)
    cites = [f"{h['section']} ({h['source']})" for h in hits]
    return format_context(hits), cites, rlog


def fallback_with_citations(summary: str, cites: list[str]) -> str:
    """Offline explanation used when no LLM key is configured."""
    if not cites:
        return summary
    return summary + "\nGuidance consulted: " + "; ".join(cites[:3])
