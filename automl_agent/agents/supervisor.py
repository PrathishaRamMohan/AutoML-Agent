"""Supervisor Agent: decides which specialist runs next and handles self-correction.

Routing is deterministic (reliable + testable); the LLM is used by the specialists
for reasoning/explanations. The supervisor can send work BACK when the Evaluation
Agent reports a problem:
  * overfitting            -> re-tune in 'regularized' mode
  * no lift over baseline  -> try the next-best model from the leaderboard
  * max retries reached    -> escalate to the human reviewer with a warning
"""
from automl_agent import config
from automl_agent.agents.common import trace

NAME = "Supervisor"


def _next_untried_model(state):
    tried = set(state.get("tried_models") or [])
    for r in state["leaderboard"]:
        if r["model"] != "baseline" and r["model"] not in tried:
            return r["model"]
    return None


def supervisor_node(state):
    approval = state.get("approval") or {}

    if not state.get("profile"):
        return {"next_agent": "profiler", "status": "running",
                "trace": trace(state, NAME, "New run: sending the dataset to the Profiler Agent.", "route")}
    if not state.get("fe_plan"):
        return {"next_agent": "feature_engineer",
                "trace": trace(state, NAME, "Profile ready → Feature Engineering Agent builds the preprocessing plan.", "route")}
    if not state.get("leaderboard"):
        return {"next_agent": "model_selector",
                "trace": trace(state, NAME, "Plan ready → Model Selection Agent compares candidates with 5-fold CV.", "route")}
    if not state.get("tuning"):
        mode = state.get("tuning_mode", "standard")
        return {"next_agent": "tuner",
                "trace": trace(state, NAME, f"Tuning Agent: optimise '{state['best_model']}' ({mode} mode).", "route")}
    if not state.get("evaluation"):
        return {"next_agent": "evaluator",
                "trace": trace(state, NAME, "Tuning done → Evaluation Agent tests on the held-out set.", "route")}

    # ---- evaluation exists: self-correction loop ----
    issues = state["evaluation"]["issues"]
    retries = state.get("retries", 0)
    remediation = list(state.get("remediation") or [])
    if not approval and issues and retries < config.MAX_RETRIES:
        if "overfitting" in issues and state.get("tuning_mode") != "regularized":
            msg = (f"Overfitting detected (gap {state['evaluation']['overfit_gap']}). "
                   f"Sending back to Tuning Agent with a regularized search space (retry {retries + 1}).")
            remediation.append(msg)
            return {"next_agent": "tuner", "tuning_mode": "regularized", "tuning": None, "evaluation": None,
                    "retries": retries + 1, "remediation": remediation,
                    "trace": trace(state, NAME, msg, "remediation")}
        if "no_lift_over_baseline" in issues:
            alt = _next_untried_model(state)
            if alt:
                msg = f"Model does not beat the baseline. Switching to next candidate '{alt}' (retry {retries + 1})."
                remediation.append(msg)
                return {"next_agent": "tuner", "best_model": alt, "tried_models": [*state["tried_models"], alt],
                        "tuning_mode": "standard", "tuning": None, "evaluation": None,
                        "retries": retries + 1, "remediation": remediation,
                        "trace": trace(state, NAME, msg, "remediation")}

    if not approval:
        warn = f" ⚠ Unresolved issues: {issues}." if issues else ""
        return {"next_agent": "human_review", "status": "awaiting_approval",
                "trace": trace(state, NAME, f"Evaluation complete. Requesting human approval.{warn}", "route")}

    decision = approval.get("decision")
    if decision == "approve" and not state.get("artifacts"):
        return {"next_agent": "finalizer",
                "trace": trace(state, NAME, f"Approved by {approval.get('reviewer')}. Finalizer exports the model.", "route")}
    if decision == "reject":
        return {"next_agent": "end", "status": "rejected",
                "trace": trace(state, NAME, f"Rejected by {approval.get('reviewer')}. Run closed.", "route")}
    return {"next_agent": "end"}


def route(state) -> str:
    return state.get("next_agent", "end")
