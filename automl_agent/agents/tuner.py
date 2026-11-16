"""Tuning Agent: Optuna hyperparameter search (standard or regularized mode)."""
from automl_agent import config
from automl_agent.agents.common import fallback_with_citations, n_classes, retrieve, trace
from automl_agent.llm import explain
from automl_agent.tools.features import train_test
from automl_agent.tools.tuning import tune

NAME = "Tuning Agent"


def tuner_node(state):
    p = state["profile"]
    model = state["best_model"]
    mode = state.get("tuning_mode", "standard")
    X_tr, _, y_tr, _, _ = train_test(state["dataset_path"], state["target"], p["task_type"])
    result = tune(model, X_tr, y_tr, p["task_type"], state["fe_plan"], n_classes(state),
                  n_trials=state.get("n_trials", config.N_TRIALS), mode=mode)

    # Guardrail: if tuning did not beat the untuned default, keep the defaults.
    untuned = next((r["cv_mean"] for r in state["leaderboard"] if r["model"] == model), None)
    if untuned is not None and result["best_cv_score"] < untuned and mode == "standard":
        result["best_params"] = {}
        result["note"] = f"Tuned score {result['best_cv_score']} < default {untuned}; kept default params."
        result["best_cv_score"] = untuned

    summary = (
        f"Model: {model}, mode: {mode}, trials: {result['n_trials']}\n"
        f"Best CV {result['metric']}: {result['best_cv_score']} (untuned: {untuned})\n"
        f"Best params: {result['best_params']}\n{result.get('note', '')}"
    )
    context, cites, rlog = retrieve(
        [f"{model.replace('_', ' ')} hyperparameter search space",
         "regularised tuning overfitting" if mode == "regularized" else "Optuna bayesian optimisation"]
    )
    message = explain(
        NAME,
        "interpret the tuned hyperparameters and what they imply about model complexity.",
        summary,
        context,
        fallback=fallback_with_citations(summary, cites),
    )
    return {"tuning": result, "trace": trace(state, NAME, message, "tuning", {"result": result, "rag": rlog})}
