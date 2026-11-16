import numpy as np
import pandas as pd

from automl_agent.rag import agentic_retrieve
from automl_agent.tools.features import build_plan, train_test
from automl_agent.tools.modeling import compare_models, pick_best
from automl_agent.tools.profiling import infer_task_type, profile_dataset


def test_task_type_inference():
    assert infer_task_type(pd.Series([0, 1, 1, 0])) == "classification"
    assert infer_task_type(pd.Series(["a", "b"])) == "classification"
    assert infer_task_type(pd.Series(np.random.rand(100))) == "regression"


def test_profiler_catches_data_problems(sample_path):
    p = profile_dataset(pd.read_csv(sample_path), "churn")
    assert p["task_type"] == "classification"
    assert "customer_id" in p["id_like_columns"]
    assert "region_code" in p["constant_columns"]
    assert "account_closed" in p["leakage_suspects"]
    assert p["columns_with_missing"]


def test_plan_drops_bad_columns(sample_path):
    plan = build_plan(profile_dataset(pd.read_csv(sample_path), "churn"))
    assert {"customer_id", "region_code", "account_closed"} <= set(plan["drop_columns"])
    assert "account_closed" not in plan["numeric_columns"]
    assert plan["use_class_weight"]


def test_models_beat_baseline(sample_path):
    p = profile_dataset(pd.read_csv(sample_path), "churn")
    plan = build_plan(p)
    X_tr, _, y_tr, _, classes = train_test(sample_path, "churn", "classification")
    assert classes == [0, 1]
    board = compare_models(X_tr, y_tr, "classification", plan, 2)
    baseline = next(r for r in board if r["model"] == "baseline")
    assert board[0]["cv_mean"] > baseline["cv_mean"] + 0.1
    assert pick_best(board) != "baseline"


def test_regression_dataset(tmp_path):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x1": rng.normal(size=300), "x2": rng.normal(size=300),
                       "cat": rng.choice(["a", "b"], 300)})
    df["y"] = 3 * df.x1 - 2 * df.x2 + (df.cat == "a") + rng.normal(0, 0.3, 300)
    p = profile_dataset(df, "y")
    assert p["task_type"] == "regression"


def test_agentic_rag_finds_relevant_docs():
    hits, log = agentic_retrieve(["target data leakage feature recorded after outcome"])
    assert any("leakage" in h["section"].lower() for h in hits)
    assert log
