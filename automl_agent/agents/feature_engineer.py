"""Feature Engineering Agent: turns the profile into a leakage-safe preprocessing plan."""
from automl_agent.agents.common import fallback_with_citations, retrieve, trace
from automl_agent.llm import explain
from automl_agent.tools.features import build_plan, summarize_plan

NAME = "Feature Engineering Agent"


def feature_engineer_node(state):
    plan = build_plan(state["profile"])
    summary = summarize_plan(plan)

    queries = ["one-hot encoding categorical handle unknown", "imputation fitted on training data only pipeline"]
    if plan["frequency_encoded_columns"]:
        queries.append("high cardinality categorical frequency encoding")
    if plan["use_class_weight"]:
        queries.append("class_weight balanced scale_pos_weight")
    context, cites, rlog = retrieve(queries)

    message = explain(
        NAME,
        "justify each preprocessing decision (dropped columns, encoding, imputation, class weights).",
        summary,
        context,
        fallback=fallback_with_citations(summary, cites),
    )
    return {"fe_plan": plan, "trace": trace(state, NAME, message, "feature_plan", {"plan": plan, "rag": rlog})}
