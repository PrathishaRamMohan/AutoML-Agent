"""Profiler Agent: inspects the dataset and flags data-quality risks."""
import pandas as pd

from automl_agent.agents.common import fallback_with_citations, retrieve, trace
from automl_agent.llm import explain
from automl_agent.tools.profiling import profile_dataset, summarize_profile

NAME = "Profiler Agent"


def profiler_node(state):
    df = pd.read_csv(state["dataset_path"])
    profile = profile_dataset(df, state["target"])
    summary = summarize_profile(profile)

    # Decide WHAT to look up based on what was found (agentic retrieval)
    queries = ["data quality checks before modelling"]
    if profile["leakage_suspects"]:
        queries.append("target data leakage feature recorded after outcome")
    if profile["columns_with_missing"]:
        queries.append("missing values imputation numeric median")
    if (profile.get("imbalance_ratio") or 1) >= 1.5:
        queries.append("class imbalance metrics and class weighting")
    if profile["id_like_columns"]:
        queries.append("identifier ID columns drop")
    context, cites, rlog = retrieve(queries)

    message = explain(
        NAME,
        "explain the most important data-quality risks and what should be done about each.",
        summary,
        context,
        fallback=fallback_with_citations(summary, cites),
    )
    return {
        "profile": profile,
        "trace": trace(state, NAME, message, "profile", {"summary": summary, "rag": rlog}),
    }
