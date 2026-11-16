"""Evaluation Agent: honest hold-out evaluation, overfitting & baseline checks."""
import joblib

from automl_agent.agents.common import fallback_with_citations, n_classes, retrieve, run_dir, trace
from automl_agent.llm import explain
from automl_agent.tools.evaluation import evaluate
from automl_agent.tools.features import train_test

NAME = "Evaluation Agent"


def evaluator_node(state):
    p = state["profile"]
    X_tr, X_te, y_tr, y_te, classes = train_test(state["dataset_path"], state["target"], p["task_type"])
    result, pipe = evaluate(state["best_model"], state["tuning"]["best_params"], X_tr, X_te, y_tr, y_te,
                            p["task_type"], state["fe_plan"], n_classes(state))
    joblib.dump({"pipeline": pipe, "classes": classes, "target": state["target"],
                 "features": list(X_tr.columns), "evaluation": result},
                run_dir(state) / "model_candidate.joblib")

    top = ", ".join(f"{f['feature']} ({f['importance']})" for f in result["feature_importance"][:5])
    summary = (
        f"Test {result['metric']}: {result['test_score']} | train: {result['train_score']} | "
        f"gap: {result['overfit_gap']} | baseline: {result['baseline_score']} | lift: {result['lift_over_baseline']}\n"
        f"Details: {result['details']}\nTop features: {top}\nIssues: {result['issues'] or 'none'}"
    )
    queries = ["hold-out test set evaluation deployment readiness", "permutation importance"]
    if "overfitting" in result["issues"]:
        queries.append("detecting overfitting remedies regularisation")
    if "no_lift_over_baseline" in result["issues"]:
        queries.append("underfitting baseline comparison")
    context, cites, rlog = retrieve(queries)
    message = explain(
        NAME,
        "state whether this model is ready for human review, and flag any risks.",
        summary,
        context,
        fallback=fallback_with_citations(summary, cites),
    )
    return {"evaluation": result, "trace": trace(state, NAME, message, "evaluation", {"result": result, "rag": rlog})}
