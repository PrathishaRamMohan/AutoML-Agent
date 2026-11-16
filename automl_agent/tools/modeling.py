"""Model factory + cross-validated comparison of candidate models."""
from __future__ import annotations

import time

import numpy as np
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier, XGBRegressor

from automl_agent import config
from automl_agent.tools.features import build_preprocessor

CLASSIFIERS = ["logistic_regression", "random_forest", "xgboost"]
REGRESSORS = ["ridge", "random_forest", "xgboost"]


def primary_metric(task_type: str, n_classes: int | None) -> str:
    """sklearn scoring name used everywhere (all on a 'higher is better' scale)."""
    if task_type == "regression":
        return "r2"
    return "roc_auc" if n_classes == 2 else "f1_macro"


def candidate_names(task_type: str) -> list[str]:
    return CLASSIFIERS if task_type == "classification" else REGRESSORS


def make_model(name: str, params: dict | None, task_type: str, plan: dict, n_classes: int | None):
    params = dict(params or {})
    rs = config.RANDOM_STATE
    cw = "balanced" if plan.get("use_class_weight") else None

    if task_type == "classification":
        if name == "logistic_regression":
            return LogisticRegression(max_iter=2000, class_weight=cw, random_state=rs, **params)
        if name == "random_forest":
            params.setdefault("n_estimators", 300)
            return RandomForestClassifier(class_weight=cw, random_state=rs, n_jobs=-1, **params)
        if name == "xgboost":
            params.setdefault("n_estimators", 300)
            params.setdefault("learning_rate", 0.1)
            params.setdefault("max_depth", 4)
            if n_classes == 2 and plan.get("use_class_weight"):
                params.setdefault("scale_pos_weight", float(plan.get("imbalance_ratio", 1.0)))
            return XGBClassifier(random_state=rs, n_jobs=-1, eval_metric="logloss",
                                 tree_method="hist", **params)
        if name == "baseline":
            return DummyClassifier(strategy="most_frequent")
    else:
        if name == "ridge":
            return Ridge(random_state=rs, **params)
        if name == "random_forest":
            params.setdefault("n_estimators", 300)
            return RandomForestRegressor(random_state=rs, n_jobs=-1, **params)
        if name == "xgboost":
            params.setdefault("n_estimators", 300)
            params.setdefault("learning_rate", 0.1)
            params.setdefault("max_depth", 4)
            return XGBRegressor(random_state=rs, n_jobs=-1, tree_method="hist", **params)
        if name == "baseline":
            return DummyRegressor(strategy="mean")
    raise ValueError(f"Unknown model '{name}' for {task_type}")


def make_pipeline(name, params, task_type, plan, n_classes) -> Pipeline:
    return Pipeline(
        [
            ("prep", build_preprocessor(plan)),
            ("model", make_model(name, params, task_type, plan, n_classes)),
        ]
    )


def cv_splitter(task_type: str):
    if task_type == "classification":
        return StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    return KFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)


def compare_models(X_train, y_train, task_type: str, plan: dict, n_classes: int | None) -> list[dict]:
    """Cross-validate every candidate (plus a dummy baseline) with identical folds."""
    metric = primary_metric(task_type, n_classes)
    cv = cv_splitter(task_type)
    board = []
    for name in candidate_names(task_type) + ["baseline"]:
        t0 = time.time()
        pipe = make_pipeline(name, None, task_type, plan, n_classes)
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring=metric, n_jobs=1)
        board.append(
            {
                "model": name,
                "metric": metric,
                "cv_mean": round(float(np.mean(scores)), 4),
                "cv_std": round(float(np.std(scores)), 4),
                "fit_seconds": round(time.time() - t0, 2),
            }
        )
    board.sort(key=lambda r: r["cv_mean"], reverse=True)
    return board


def pick_best(board: list[dict]) -> str:
    """Best score wins, but prefer a simpler model if it is within one std (Occam's razor)."""
    candidates = [r for r in board if r["model"] != "baseline"]
    top = candidates[0]
    simplicity = {"logistic_regression": 0, "ridge": 0, "random_forest": 1, "xgboost": 2}
    close = [r for r in candidates if top["cv_mean"] - r["cv_mean"] <= top["cv_std"]]
    close.sort(key=lambda r: (simplicity.get(r["model"], 9), -r["cv_mean"]))
    return close[0]["model"]
