"""Supervisor routing + self-correction logic (no ML needed)."""
from automl_agent.agents.supervisor import supervisor_node

BASE = {
    "run_id": "test-sup",
    "profile": {"task_type": "classification"},
    "fe_plan": {"drop_columns": {}},
    "leaderboard": [
        {"model": "xgboost", "cv_mean": 0.8}, {"model": "random_forest", "cv_mean": 0.79},
        {"model": "logistic_regression", "cv_mean": 0.7}, {"model": "baseline", "cv_mean": 0.5},
    ],
    "best_model": "xgboost",
    "tried_models": ["xgboost"],
    "tuning": {"best_params": {}},
    "retries": 0,
    "tuning_mode": "standard",
}


def ev(*issues):
    return {"issues": list(issues), "overfit_gap": 0.2}


def test_routes_in_order():
    assert supervisor_node({"run_id": "t"})["next_agent"] == "profiler"
    assert supervisor_node({**BASE, "tuning": None})["next_agent"] == "tuner"
    assert supervisor_node({**BASE, "evaluation": None})["next_agent"] == "evaluator"


def test_overfitting_triggers_regularized_retune():
    out = supervisor_node({**BASE, "evaluation": ev("overfitting")})
    assert out["next_agent"] == "tuner"
    assert out["tuning_mode"] == "regularized"
    assert out["retries"] == 1 and out["tuning"] is None


def test_no_lift_switches_model():
    out = supervisor_node({**BASE, "evaluation": ev("no_lift_over_baseline")})
    assert out["next_agent"] == "tuner"
    assert out["best_model"] == "random_forest"


def test_escalates_to_human_after_max_retries():
    out = supervisor_node({**BASE, "retries": 2, "evaluation": ev("overfitting")})
    assert out["next_agent"] == "human_review"


def test_clean_eval_goes_to_human_then_finalizer():
    assert supervisor_node({**BASE, "evaluation": ev()})["next_agent"] == "human_review"
    approved = {**BASE, "evaluation": ev(), "approval": {"decision": "approve", "reviewer": "A"}}
    assert supervisor_node(approved)["next_agent"] == "finalizer"
    rejected = {**BASE, "evaluation": ev(), "approval": {"decision": "reject", "reviewer": "A"}}
    assert supervisor_node(rejected)["next_agent"] == "end"
