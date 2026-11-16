"""LangGraph wiring: Supervisor ⇄ specialist agents, with a human-approval interrupt.

            ┌──────────── Supervisor ────────────┐
            │   (routes + self-correction loop)  │
   profiler → feature_engineer → model_selector → tuner → evaluator
            │                                    │
            └── human_review (interrupt) → finalizer → END
"""
from __future__ import annotations

import sqlite3
import uuid
from functools import lru_cache

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from automl_agent import config, db
from automl_agent.agents import (
    evaluator_node,
    feature_engineer_node,
    finalizer_node,
    human_review_node,
    model_selector_node,
    profiler_node,
    route,
    supervisor_node,
    tuner_node,
)
from automl_agent.state import AutoMLState

WORKERS = {
    "profiler": profiler_node,
    "feature_engineer": feature_engineer_node,
    "model_selector": model_selector_node,
    "tuner": tuner_node,
    "evaluator": evaluator_node,
    "human_review": human_review_node,
}


def build_graph(checkpointer=None):
    g = StateGraph(AutoMLState)
    g.add_node("supervisor", supervisor_node)
    for name, fn in WORKERS.items():
        g.add_node(name, fn)
        g.add_edge(name, "supervisor")          # every specialist reports back
    g.add_node("finalizer", finalizer_node)

    g.add_edge(START, "supervisor")
    g.add_conditional_edges(
        "supervisor",
        route,
        {**{n: n for n in WORKERS}, "finalizer": "finalizer", "end": END},
    )
    g.add_edge("finalizer", END)
    return g.compile(checkpointer=checkpointer)


@lru_cache(maxsize=1)
def get_app():
    """Graph compiled with a persistent SQLite checkpointer (so paused runs survive restarts)."""
    conn = sqlite3.connect(config.CHECKPOINT_DB, check_same_thread=False)
    return build_graph(SqliteSaver(conn))


# ---------------------------------------------------------------- convenience API
def new_run(dataset_path: str, target: str, goal: str = "", n_trials: int | None = None) -> tuple[str, dict]:
    run_id = uuid.uuid4().hex[:8]
    db.create_run(run_id, str(dataset_path), target, goal)
    inputs: AutoMLState = {
        "run_id": run_id, "dataset_path": str(dataset_path), "target": target, "goal": goal,
        "n_trials": n_trials or config.N_TRIALS, "tuning_mode": "standard", "retries": 0,
        "remediation": [], "tried_models": [], "trace": [],
    }
    return run_id, inputs


def thread(run_id: str) -> dict:
    return {"configurable": {"thread_id": run_id}, "recursion_limit": 60}


def stream(inputs_or_command, run_id: str):
    """Yield (node_name, update) as each agent finishes. Handles interrupts."""
    for chunk in get_app().stream(inputs_or_command, thread(run_id), stream_mode="updates"):
        for node, update in chunk.items():
            yield node, update


def resume_command(decision: dict) -> Command:
    return Command(resume=decision)


def get_state(run_id: str) -> dict:
    return get_app().get_state(thread(run_id)).values


def pending_interrupt(run_id: str):
    snap = get_app().get_state(thread(run_id))
    for task in snap.tasks:
        if task.interrupts:
            return task.interrupts[0].value
    return None


def run_until_pause(dataset_path, target, goal="", n_trials=None) -> str:
    """Run all agents until the human-approval checkpoint. Returns run_id."""
    run_id, inputs = new_run(dataset_path, target, goal, n_trials)
    for _ in stream(inputs, run_id):
        pass
    db.update_run(run_id, status=get_state(run_id).get("status", "running"),
                  task_type=get_state(run_id)["profile"]["task_type"])
    return run_id


def decide(run_id: str, decision: dict) -> dict:
    for _ in stream(resume_command(decision), run_id):
        pass
    state = get_state(run_id)
    if state.get("status") == "rejected":
        db.update_run(run_id, status="rejected")
    elif state.get("status") == "awaiting_approval":
        db.update_run(run_id, status="awaiting_approval")
    return state
