"""Central configuration. Values come from environment variables / .env file."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

RUNS_DIR = ROOT_DIR / "runs"
KB_DIR = ROOT_DIR / "knowledge_base"
DATA_DIR = ROOT_DIR / "data"
DB_PATH = RUNS_DIR / "automl.sqlite"
CHECKPOINT_DB = RUNS_DIR / "checkpoints.sqlite"

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
USE_LLM = os.getenv("USE_LLM", "true").lower() == "true" and bool(GROQ_API_KEY)
EMBEDDINGS = os.getenv("EMBEDDINGS", "tfidf").lower()
N_TRIALS = int(os.getenv("N_TRIALS", "25"))

# Agent behaviour
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
MAX_RETRIES = 2            # how many times the supervisor may send work back for remediation
OVERFIT_GAP = 0.10         # train-vs-test gap that counts as overfitting
MIN_LIFT_OVER_BASELINE = 0.02

RUNS_DIR.mkdir(parents=True, exist_ok=True)
