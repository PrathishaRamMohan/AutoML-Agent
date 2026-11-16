"""Data profiling tool: inspects a dataset and flags quality problems.

Pure Python / pandas — no LLM. The Profiler Agent calls this and then
asks the LLM to explain the findings using RAG context.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def infer_task_type(y: pd.Series) -> str:
    """Return 'classification' or 'regression'."""
    if y.dtype == object or str(y.dtype) in ("category", "bool", "string", "str"):
        return "classification"
    n_unique = y.nunique(dropna=True)
    if pd.api.types.is_integer_dtype(y) and n_unique <= 20:
        return "classification"
    if pd.api.types.is_float_dtype(y) and n_unique <= 10 and np.allclose(y.dropna() % 1, 0):
        return "classification"
    return "regression"


def _is_datetime_like(s: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(s):
        return True
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        sample = s.dropna().astype(str).head(50)
        if len(sample) == 0:
            return False
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        return parsed.notna().mean() > 0.9 and sample.str.contains(r"[-/:]").mean() > 0.9
    return False


def _single_feature_score(x: pd.Series, y: pd.Series, task_type: str) -> float | None:
    """How well does ONE feature predict the target on its own? (leakage check)"""
    mask = x.notna() & y.notna()
    x, y = x[mask], y[mask]
    if len(x) < 20 or x.nunique() < 2:
        return None
    if task_type == "classification":
        if y.nunique() != 2:
            return None
        y_bin = (y == sorted(y.unique())[1]).astype(int)
        if pd.api.types.is_numeric_dtype(x):
            auc = roc_auc_score(y_bin, x)
        else:  # categorical: use per-category target mean as a score
            auc = roc_auc_score(y_bin, x.map(y_bin.groupby(x).mean()))
        return float(max(auc, 1 - auc))
    if pd.api.types.is_numeric_dtype(x):
        return float(abs(np.corrcoef(x, y)[0, 1]))
    return None


def profile_dataset(df: pd.DataFrame, target: str) -> dict:
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found. Columns: {list(df.columns)}")

    y = df[target]
    X = df.drop(columns=[target])
    task_type = infer_task_type(y)
    n_rows = len(df)

    columns = {}
    id_like, constant, high_card, datetime_cols, leakage = [], [], [], [], []
    for col in X.columns:
        s = X[col]
        n_unique = int(s.nunique(dropna=True))
        info = {
            "dtype": str(s.dtype),
            "missing_pct": round(float(s.isna().mean() * 100), 2),
            "n_unique": n_unique,
            "kind": "numeric" if pd.api.types.is_numeric_dtype(s) else "categorical",
        }
        if n_unique <= 1:
            constant.append(col)
        elif _is_datetime_like(s):
            info["kind"] = "datetime"
            datetime_cols.append(col)
        elif n_unique == n_rows or (info["kind"] == "categorical" and n_unique > 0.9 * n_rows):
            id_like.append(col)
        elif info["kind"] == "categorical" and n_unique > 50:
            high_card.append(col)

        if col not in id_like + constant + datetime_cols:
            score = _single_feature_score(s, y, task_type)
            if score is not None:
                info["single_feature_score"] = round(score, 4)
                if score >= 0.97:
                    leakage.append(col)
        columns[col] = info

    profile = {
        "n_rows": n_rows,
        "n_features": X.shape[1],
        "target": target,
        "task_type": task_type,
        "duplicate_rows": int(df.duplicated().sum()),
        "target_missing": int(y.isna().sum()),
        "columns": columns,
        "id_like_columns": id_like,
        "constant_columns": constant,
        "high_cardinality_columns": high_card,
        "datetime_columns": datetime_cols,
        "leakage_suspects": leakage,
        "columns_with_missing": {c: v["missing_pct"] for c, v in columns.items() if v["missing_pct"] > 0},
    }

    if task_type == "classification":
        counts = y.value_counts()
        profile["class_counts"] = {str(k): int(v) for k, v in counts.items()}
        profile["n_classes"] = int(len(counts))
        profile["imbalance_ratio"] = round(float(counts.max() / counts.min()), 2)
    else:
        profile["target_stats"] = {k: round(float(v), 4) for k, v in y.describe().items()}
        profile["imbalance_ratio"] = None

    return profile


def summarize_profile(p: dict) -> str:
    """Short, human-readable summary (also used as LLM input)."""
    lines = [
        f"Rows: {p['n_rows']}, features: {p['n_features']}, task: {p['task_type']}",
        f"Duplicate rows: {p['duplicate_rows']}",
    ]
    if p["task_type"] == "classification":
        lines.append(f"Class counts: {p['class_counts']} (imbalance ratio {p['imbalance_ratio']}:1)")
    if p["columns_with_missing"]:
        lines.append(f"Missing values (%): {p['columns_with_missing']}")
    for key, label in [
        ("id_like_columns", "ID-like columns"),
        ("constant_columns", "Constant columns"),
        ("high_cardinality_columns", "High-cardinality categoricals"),
        ("datetime_columns", "Datetime columns"),
        ("leakage_suspects", "LEAKAGE suspects"),
    ]:
        if p[key]:
            lines.append(f"{label}: {p[key]}")
    return "\n".join(lines)
