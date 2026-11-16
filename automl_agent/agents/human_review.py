"""Human-in-the-loop checkpoint.

`interrupt()` pauses the LangGraph run and persists state in the SQLite checkpointer.
The UI / API / CLI resumes it with  Command(resume={...decision...}).

Accepted decisions:
  {"decision": "approve", "reviewer": "...", "comment": "..."}
  {"decision": "reject",  "reviewer": "...", "comment": "..."}
  {"decision": "retune",  "reviewer": "...", "n_trials": 50}   -> run tuning again with more trials
"""
from langgraph.types import interrupt

from automl_agent import db
from automl_agent.agents.common import trace

NAME = "Human Reviewer"


def human_review_node(state):
    ev = state["evaluation"]
    decision = interrupt(
        {
            "question": "Approve this model for export?",
            "model": ev["model"],
            "metric": ev["metric"],
            "test_score": ev["test_score"],
            "baseline_score": ev["baseline_score"],
            "issues": ev["issues"],
            "options": ["approve", "reject", "retune"],
        }
    )
    decision = dict(decision or {})
    decision.setdefault("reviewer", "anonymous")
    decision.setdefault("comment", "")
    decision["timestamp"] = db.now()
    choice = decision.get("decision")

    msg = f"{decision['reviewer']} chose '{choice}'." + (f" Comment: {decision['comment']}" if decision["comment"] else "")
    db.update_run(state["run_id"], reviewer=decision["reviewer"], decision=choice)

    if choice == "retune":
        n = int(decision.get("n_trials") or state.get("n_trials", 25) * 2)
        return {"approval": None, "tuning": None, "evaluation": None, "n_trials": n, "status": "running",
                "trace": trace(state, NAME, msg + f" Re-tuning with {n} trials.", "human_decision", decision)}
    if choice not in ("approve", "reject"):
        raise ValueError(f"Unknown decision: {choice}")
    return {"approval": decision, "trace": trace(state, NAME, msg, "human_decision", decision)}
