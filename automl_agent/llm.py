"""Thin wrapper around Groq-hosted LLaMA 3.3 70B (via langchain-groq).

If no GROQ_API_KEY is set (or USE_LLM=false) every call returns the provided
`fallback` text, so the whole system still runs offline for demos and tests.
"""
from __future__ import annotations

from functools import lru_cache

from automl_agent import config

SYSTEM_PROMPT = (
    "You are one agent inside an AutoML multi-agent system. Be concise, factual and specific. "
    "Ground your statements in the provided tool results and knowledge-base context; cite context "
    "blocks like [1], [2]. Never invent numbers that are not in the tool results."
)


@lru_cache(maxsize=1)
def _client():
    from langchain_groq import ChatGroq

    return ChatGroq(model=config.GROQ_MODEL, api_key=config.GROQ_API_KEY, temperature=0.2, max_tokens=600)


def llm_available() -> bool:
    return config.USE_LLM


def ask_llm(instruction: str, content: str, fallback: str = "") -> str:
    if not config.USE_LLM:
        return fallback
    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        resp = _client().invoke(
            [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=f"{instruction}\n\n{content}")]
        )
        return resp.content.strip()
    except Exception as exc:  # network / quota problems should never crash the pipeline
        return f"{fallback}\n\n(LLM unavailable: {type(exc).__name__})" if fallback else ""


def explain(agent: str, task: str, tool_output: str, rag_context: str, fallback: str) -> str:
    """Standard pattern used by every agent: tool output + retrieved context -> explanation."""
    return ask_llm(
        f"As the {agent}, {task} Use at most 6 bullet points.",
        f"TOOL RESULTS:\n{tool_output}\n\nKNOWLEDGE BASE CONTEXT:\n{rag_context}",
        fallback=fallback,
    )
