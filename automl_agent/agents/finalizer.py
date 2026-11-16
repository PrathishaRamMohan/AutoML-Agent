"""Finalizer Agent: runs only after human approval. Exports model + PDF model card, registers run."""
import shutil

from automl_agent import db
from automl_agent.agents.common import run_dir, trace
from automl_agent.report import build_report

NAME = "Finalizer Agent"


def finalizer_node(state):
    d = run_dir(state)
    model_path = d / "model.joblib"
    shutil.copy(d / "model_candidate.joblib", model_path)

    # the report should include the finalizer's own step, so pass the updated trace
    note = trace(state, NAME, f"Exported approved model to {model_path.name} and generated the model card PDF.", "export")
    report_path = build_report({**state, "trace": state.get("trace", []) + note}, d / "model_card.pdf")

    ev = state["evaluation"]
    db.update_run(state["run_id"], status="approved", best_model=ev["model"], metric=ev["metric"],
                  test_score=ev["test_score"], model_path=str(model_path), report_path=str(report_path))
    return {"artifacts": {"model": str(model_path), "report": str(report_path)}, "status": "approved",
            "next_agent": "end", "trace": note}
