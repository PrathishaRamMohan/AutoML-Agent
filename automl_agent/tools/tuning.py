"""Hyperparameter tuning with Optuna (TPE). Two modes: 'standard' and 'regularized'
(used when the Evaluator detects overfitting and the Supervisor sends the job back)."""
from __future__ import annotations

import numpy as np
import optuna
from sklearn.model_selection import cross_val_score

from automl_agent.tools.modeling import cv_splitter, make_pipeline, primary_metric

optuna.logging.set_verbosity(optuna.logging.WARNING)


def suggest_params(trial: optuna.Trial, name: str, mode: str) -> dict:
    reg = mode == "regularized"
    if name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "max_depth": trial.suggest_int("max_depth", 2, 4 if reg else 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1 if reg else 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 0.9 if reg else 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 3 if reg else 1, 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0 if reg else 1e-3, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        }
    if name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 8 if reg else 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5 if reg else 1, 20 if reg else 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
        }
    if name == "logistic_regression":
        return {"C": trial.suggest_float("C", 1e-3, 1.0 if reg else 100.0, log=True)}
    if name == "ridge":
        return {"alpha": trial.suggest_float("alpha", 1.0 if reg else 1e-3, 100.0, log=True)}
    raise ValueError(name)


def tune(name: str, X_train, y_train, task_type: str, plan: dict, n_classes: int | None,
         n_trials: int = 25, mode: str = "standard", timeout: int = 300) -> dict:
    metric = primary_metric(task_type, n_classes)
    cv = cv_splitter(task_type)

    def objective(trial):
        params = suggest_params(trial, name, mode)
        pipe = make_pipeline(name, params, task_type, plan, n_classes)
        return float(np.mean(cross_val_score(pipe, X_train, y_train, cv=cv, scoring=metric)))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    history = [round(t.value, 4) for t in study.trials if t.value is not None]
    best_so_far = list(np.maximum.accumulate(history)) if history else []
    return {
        "model": name,
        "mode": mode,
        "metric": metric,
        "n_trials": len(study.trials),
        "best_cv_score": round(float(study.best_value), 4),
        "best_params": study.best_params,
        "trial_scores": history,
        "best_so_far": [round(float(v), 4) for v in best_so_far],
    }
