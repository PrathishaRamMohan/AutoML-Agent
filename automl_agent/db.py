"""SQLite persistence: run registry + audit log (every agent action and human decision)."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime

from automl_agent import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    created_at TEXT,
    dataset TEXT,
    target TEXT,
    goal TEXT,
    task_type TEXT,
    status TEXT,
    best_model TEXT,
    metric TEXT,
    test_score REAL,
    reviewer TEXT,
    decision TEXT,
    model_path TEXT,
    report_path TEXT
);
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    ts TEXT,
    agent TEXT,
    event TEXT,
    detail TEXT
);
"""


@contextmanager
def connect():
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def create_run(run_id: str, dataset: str, target: str, goal: str) -> None:
    with connect() as c:
        c.execute(
            "INSERT OR REPLACE INTO runs (run_id, created_at, dataset, target, goal, status) VALUES (?,?,?,?,?,?)",
            (run_id, now(), dataset, target, goal, "running"),
        )


def update_run(run_id: str, **fields) -> None:
    if not fields:
        return
    cols = ", ".join(f"{k}=?" for k in fields)
    with connect() as c:
        c.execute(f"UPDATE runs SET {cols} WHERE run_id=?", (*fields.values(), run_id))


def log(run_id: str, agent: str, event: str, detail=None) -> None:
    if not isinstance(detail, str):
        detail = json.dumps(detail, default=str)
    with connect() as c:
        c.execute(
            "INSERT INTO audit_log (run_id, ts, agent, event, detail) VALUES (?,?,?,?,?)",
            (run_id, now(), agent, event, detail),
        )


def list_runs() -> list[dict]:
    with connect() as c:
        return [dict(r) for r in c.execute("SELECT * FROM runs ORDER BY created_at DESC")]


def get_run(run_id: str) -> dict | None:
    with connect() as c:
        r = c.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        return dict(r) if r else None


def get_audit(run_id: str) -> list[dict]:
    with connect() as c:
        return [dict(r) for r in c.execute("SELECT * FROM audit_log WHERE run_id=? ORDER BY id", (run_id,))]
