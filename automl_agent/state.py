"""Shared LangGraph state passed between all agents.

Everything here is JSON-serialisable so the SQLite checkpointer can persist it
(DataFrames and fitted models are saved to runs/<run_id>/ instead).
"""
from __future__ import annotations

import operator
from typing import Annotated, Optional, TypedDict


class AutoMLState(TypedDict, total=False):
    # --- inputs ---
    run_id: str
    dataset_path: str
    target: str
    goal: str
    n_trials: int

    # --- produced by agents ---
    profile: Optional[dict]          # Profiler Agent
    fe_plan: Optional[dict]          # Feature Engineering Agent
    leaderboard: Optional[list]      # Model Selection Agent
    best_model: Optional[str]
    tried_models: list
    tuning: Optional[dict]           # Tuning Agent
    tuning_mode: str                 # "standard" | "regularized"
    evaluation: Optional[dict]       # Evaluation Agent

    # --- supervisor bookkeeping ---
    next_agent: str
    retries: int
    remediation: list                # history of self-corrections
    approval: Optional[dict]         # Human-in-the-loop decision
    artifacts: Optional[dict]        # Finalizer outputs (model + report paths)
    status: str

    # --- append-only reasoning trace (each agent adds entries) ---
    trace: Annotated[list, operator.add]
