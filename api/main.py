"""FastAPI backend.

    uvicorn api.main:app --reload --port 8000
    Docs: http://localhost:8000/docs
"""
from __future__ import annotations

import shutil
import uuid
from typing import Literal, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from automl_agent import config, db
from automl_agent.graph import decide, get_state, pending_interrupt, run_until_pause

app = FastAPI(title="AutoML-Agent API", version="1.0",
              description="Multi-agent AutoML (LangGraph) with human-in-the-loop approval.")


class Decision(BaseModel):
    decision: Literal["approve", "reject", "retune"]
    reviewer: str = "api-user"
    comment: str = ""
    n_trials: Optional[int] = None


def _summary(run_id: str) -> dict:
    s = get_state(run_id)
    if not s:
        raise HTTPException(404, "Run not found")
    return {
        "run_id": run_id,
        "status": s.get("status"),
        "pending_approval": pending_interrupt(run_id),
        "best_model": s.get("best_model"),
        "leaderboard": s.get("leaderboard"),
        "evaluation": s.get("evaluation"),
        "remediation": s.get("remediation"),
        "artifacts": s.get("artifacts"),
        "trace": s.get("trace"),
    }


@app.get("/health")
def health():
    return {"status": "ok", "llm": config.USE_LLM}


@app.post("/runs")
def create_run(file: UploadFile = File(...), target: str = Form(...), goal: str = Form(""),
               n_trials: int = Form(config.N_TRIALS)):
    """Upload a CSV and run all agents until the human-approval checkpoint."""
    upload_dir = config.RUNS_DIR / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    path = upload_dir / f"{uuid.uuid4().hex[:6]}_{file.filename}"
    with path.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    if target not in pd.read_csv(path, nrows=5).columns:
        raise HTTPException(400, f"Target column '{target}' not in CSV")
    run_id = run_until_pause(str(path), target, goal, n_trials)
    return _summary(run_id)


@app.get("/runs")
def list_runs():
    return db.list_runs()


@app.get("/runs/{run_id}")
def get_run(run_id: str):
    return _summary(run_id)


@app.get("/runs/{run_id}/audit")
def audit(run_id: str):
    return db.get_audit(run_id)


@app.post("/runs/{run_id}/decision")
def post_decision(run_id: str, d: Decision):
    if pending_interrupt(run_id) is None:
        raise HTTPException(409, "This run is not waiting for approval")
    decide(run_id, d.model_dump(exclude_none=True))
    return _summary(run_id)


@app.get("/runs/{run_id}/report")
def report(run_id: str):
    s = get_state(run_id)
    if not s or not s.get("artifacts"):
        raise HTTPException(404, "Report available only after approval")
    return FileResponse(s["artifacts"]["report"], media_type="application/pdf", filename=f"model_card_{run_id}.pdf")


@app.post("/runs/{run_id}/predict")
def predict(run_id: str, rows: list[dict]):
    """Score new rows with an APPROVED model."""
    s = get_state(run_id)
    if not s or not s.get("artifacts"):
        raise HTTPException(404, "Model not approved/exported yet")
    bundle = joblib.load(s["artifacts"]["model"])
    X = pd.DataFrame(rows).reindex(columns=bundle["features"])
    pipe = bundle["pipeline"]
    preds = pipe.predict(X).tolist()
    out = {"predictions": preds}
    if bundle["classes"] is not None:
        out["labels"] = [bundle["classes"][int(p)] for p in preds]
        if hasattr(pipe, "predict_proba"):
            out["probabilities"] = pipe.predict_proba(X).round(4).tolist()
    return out
