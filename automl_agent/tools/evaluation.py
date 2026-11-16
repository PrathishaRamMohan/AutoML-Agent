"""Final evaluation on the untouched hold-out test set."""
from __future__ import annotations

import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    get_scorer,
    mean_absolute_error,
    precision_score,
    recall_score,
    r2_score,
    root_mean_squared_error,
    roc_auc_score,
)

from automl_agent import config
from automl_agent.tools.modeling import make_pipeline, primary_metric


def _score(pipe, X, y, metric):
    return float(get_scorer(metric)(pipe, X, y))


def evaluate(name, params, X_train, X_test, y_train, y_test, task_type, plan, n_classes):
    metric = primary_metric(task_type, n_classes)

    pipe = make_pipeline(name, params, task_type, plan, n_classes)
    pipe.fit(X_train, y_train)
    baseline = make_pipeline("baseline", None, task_type, plan, n_classes).fit(X_train, y_train)

    train_score = _score(pipe, X_train, y_train, metric)
    test_score = _score(pipe, X_test, y_test, metric)
    base_score = _score(baseline, X_test, y_test, metric)

    y_pred = pipe.predict(X_test)
    details: dict = {}
    if task_type == "classification":
        avg = "binary" if n_classes == 2 else "macro"
        details = {
            "accuracy": accuracy_score(y_test, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average=avg, zero_division=0),
            "recall": recall_score(y_test, y_pred, average=avg, zero_division=0),
            "f1": f1_score(y_test, y_pred, average=avg, zero_division=0),
        }
        if n_classes == 2:
            details["roc_auc"] = roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])
        details = {k: round(float(v), 4) for k, v in details.items()}
        details["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()
    else:
        details = {
            "r2": round(float(r2_score(y_test, y_pred)), 4),
            "rmse": round(float(root_mean_squared_error(y_test, y_pred)), 4),
            "mae": round(float(mean_absolute_error(y_test, y_pred)), 4),
        }

    # Permutation importance on raw input columns (model-agnostic, computed on test data)
    pi = permutation_importance(
        pipe, X_test, y_test, scoring=metric, n_repeats=5, random_state=config.RANDOM_STATE, n_jobs=1
    )
    importances = sorted(
        [
            {"feature": col, "importance": round(float(m), 4), "std": round(float(s), 4)}
            for col, m, s in zip(X_test.columns, pi.importances_mean, pi.importances_std)
            if col not in plan["drop_columns"]
        ],
        key=lambda r: r["importance"],
        reverse=True,
    )

    gap = train_score - test_score
    lift = test_score - base_score
    issues = []
    if gap > config.OVERFIT_GAP:
        issues.append("overfitting")
    if lift < config.MIN_LIFT_OVER_BASELINE:
        issues.append("no_lift_over_baseline")
    if task_type == "classification" and test_score > 0.995:
        issues.append("suspiciously_perfect")

    return {
        "model": name,
        "params": params,
        "metric": metric,
        "train_score": round(train_score, 4),
        "test_score": round(test_score, 4),
        "baseline_score": round(base_score, 4),
        "overfit_gap": round(gap, 4),
        "lift_over_baseline": round(lift, 4),
        "details": details,
        "feature_importance": importances[:15],
        "issues": issues,
    }, pipe
