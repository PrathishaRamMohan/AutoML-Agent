"""End-to-end: full LangGraph run with human-in-the-loop, plus the FastAPI endpoints."""
from pathlib import Path

from fastapi.testclient import TestClient

from automl_agent import db
from automl_agent.graph import decide, get_state, pending_interrupt, run_until_pause


def test_full_run_pauses_then_approves(sample_path):
    run_id = run_until_pause(sample_path, "churn", "test", n_trials=4)
    req = pending_interrupt(run_id)
    assert req is not None and req["options"] == ["approve", "reject", "retune"]

    state = decide(run_id, {"decision": "approve", "reviewer": "pytest"})
    assert state["status"] == "approved"
    assert Path(state["artifacts"]["model"]).exists()
    assert Path(state["artifacts"]["report"]).stat().st_size > 10_000
    assert db.get_run(run_id)["status"] == "approved"
    agents = {t["agent"] for t in state["trace"]}
    assert {"Supervisor", "Profiler Agent", "Evaluation Agent", "Human Reviewer", "Finalizer Agent"} <= agents


def test_retune_then_reject(sample_path):
    run_id = run_until_pause(sample_path, "churn", n_trials=3)
    decide(run_id, {"decision": "retune", "reviewer": "pytest", "n_trials": 4})
    assert pending_interrupt(run_id) is not None          # paused again after re-tuning
    assert get_state(run_id)["n_trials"] == 4
    state = decide(run_id, {"decision": "reject", "reviewer": "pytest"})
    assert state["status"] == "rejected" and not state.get("artifacts")


def test_api_roundtrip(sample_path):
    from api.main import app

    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"
    with open(sample_path, "rb") as f:
        r = client.post("/runs", files={"file": ("churn.csv", f, "text/csv")},
                        data={"target": "churn", "n_trials": "3"})
    assert r.status_code == 200, r.text
    run_id = r.json()["run_id"]
    assert r.json()["pending_approval"] is not None

    r = client.post(f"/runs/{run_id}/decision", json={"decision": "approve", "reviewer": "api"})
    assert r.json()["status"] == "approved"
    assert client.get(f"/runs/{run_id}/report").headers["content-type"] == "application/pdf"

    row = {"tenure_months": 2, "contract_type": "Month-to-month", "internet_service": "Fiber optic",
           "payment_method": "Electronic check", "monthly_charges": 95, "support_calls": 5,
           "senior_citizen": 1, "data_usage_gb": 10}
    pred = client.post(f"/runs/{run_id}/predict", json=[row]).json()
    assert pred["labels"][0] in (0, 1)
    assert len(client.get(f"/runs/{run_id}/audit").json()) > 5
