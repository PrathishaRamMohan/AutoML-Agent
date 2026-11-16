"""Model Selection Agent: cross-validates candidate models and picks one to tune."""
from automl_agent.agents.common import fallback_with_citations, n_classes, retrieve, trace
from automl_agent.llm import explain
from automl_agent.tools.features import train_test
from automl_agent.tools.modeling import compare_models, pick_best

NAME = "Model Selection Agent"


def model_selector_node(state):
    p = state["profile"]
    X_tr, _, y_tr, _, _ = train_test(state["dataset_path"], state["target"], p["task_type"])
    board = compare_models(X_tr, y_tr, p["task_type"], state["fe_plan"], n_classes(state))
    best = pick_best(board)

    table = "\n".join(f"{r['model']:<20} {r['metric']}={r['cv_mean']:.4f} ± {r['cv_std']:.4f}" for r in board)
    summary = f"5-fold CV leaderboard:\n{table}\nSelected for tuning: {best}"
    context, cites, rlog = retrieve(
        ["comparing models fairly cross-validation simpler model", f"{best.replace('_', ' ')} strengths"]
    )
    message = explain(
        NAME,
        "explain the leaderboard and why the selected model is the right one to tune.",
        summary,
        context,
        fallback=fallback_with_citations(summary, cites),
    )
    return {
        "leaderboard": board,
        "best_model": best,
        "tried_models": [best],
        "trace": trace(state, NAME, message, "leaderboard", {"leaderboard": board, "rag": rlog}),
    }
