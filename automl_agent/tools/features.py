"""Feature engineering tool: turns a profile into a preprocessing plan + sklearn transformer."""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from automl_agent import config


def build_plan(profile: dict) -> dict:
    """Decide which columns to drop / how to encode. Returns a JSON-serialisable plan."""
    drop = {}
    for c in profile["id_like_columns"]:
        drop[c] = "ID-like (unique per row)"
    for c in profile["constant_columns"]:
        drop[c] = "constant (zero variance)"
    for c in profile["leakage_suspects"]:
        drop[c] = "suspected target leakage"
    for c in profile["datetime_columns"]:
        drop[c] = "raw datetime (not engineered in this version)"
    for c, pct in profile["columns_with_missing"].items():
        if pct > 60 and c not in drop:
            drop[c] = f"too many missing values ({pct}%)"

    numeric, categorical, frequency = [], [], []
    for col, info in profile["columns"].items():
        if col in drop:
            continue
        if info["kind"] == "numeric":
            numeric.append(col)
        elif col in profile["high_cardinality_columns"]:
            frequency.append(col)
        else:
            categorical.append(col)

    imbalance = profile.get("imbalance_ratio") or 1.0
    return {
        "drop_columns": drop,
        "numeric_columns": numeric,
        "categorical_columns": categorical,
        "frequency_encoded_columns": frequency,
        "numeric_imputer": "median",
        "categorical_imputer": "most_frequent",
        "scale_numeric": True,
        "use_class_weight": profile["task_type"] == "classification" and imbalance >= 1.5,
        "imbalance_ratio": imbalance,
    }


def _frequency_encode(X: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(X)
    out = pd.DataFrame(index=X.index)
    for c in X.columns:
        freq = X[c].value_counts(normalize=True)
        out[c] = X[c].map(freq).fillna(0.0).astype(float)
    return out


def build_preprocessor(plan: dict) -> ColumnTransformer:
    transformers = []
    if plan["numeric_columns"]:
        steps = [("impute", SimpleImputer(strategy=plan["numeric_imputer"]))]
        if plan["scale_numeric"]:
            steps.append(("scale", StandardScaler()))
        transformers.append(("num", Pipeline(steps), plan["numeric_columns"]))
    if plan["categorical_columns"]:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy=plan["categorical_imputer"])),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                plan["categorical_columns"],
            )
        )
    if plan["frequency_encoded_columns"]:
        transformers.append(
            ("freq", FunctionTransformer(_frequency_encode), plan["frequency_encoded_columns"])
        )
    if not transformers:
        raise ValueError("No usable feature columns left after preprocessing plan.")
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def load_xy(dataset_path: str, target: str):
    df = pd.read_csv(dataset_path)
    df = df[df[target].notna()].reset_index(drop=True)
    return df.drop(columns=[target]), df[target]


def train_test(dataset_path: str, target: str, task_type: str):
    """The single, fixed train/test split used by every agent (test set is never tuned on).

    Returns X_train, X_test, y_train, y_test, classes
    (for classification, y is label-encoded to 0..k-1 and `classes` holds the original labels).
    """
    X, y = load_xy(dataset_path, target)
    classes = None
    if task_type == "classification":
        classes = sorted(y.unique().tolist(), key=str)
        mapping = {c: i for i, c in enumerate(classes)}
        y = y.map(mapping).astype(int)
        classes = [c.item() if hasattr(c, "item") else c for c in classes]
    stratify = y if task_type == "classification" and y.value_counts().min() >= 2 else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=stratify
    )
    return X_tr, X_te, y_tr, y_te, classes


def summarize_plan(plan: dict) -> str:
    lines = [
        f"Numeric ({len(plan['numeric_columns'])}): {plan['numeric_columns']}",
        f"Categorical one-hot ({len(plan['categorical_columns'])}): {plan['categorical_columns']}",
    ]
    if plan["frequency_encoded_columns"]:
        lines.append(f"Frequency-encoded: {plan['frequency_encoded_columns']}")
    if plan["drop_columns"]:
        lines.append("Dropped: " + "; ".join(f"{k} ({v})" for k, v in plan["drop_columns"].items()))
    lines.append(f"Imputation: numeric={plan['numeric_imputer']}, categorical={plan['categorical_imputer']}")
    if plan["use_class_weight"]:
        lines.append(f"Class weighting ON (imbalance {plan['imbalance_ratio']}:1)")
    return "\n".join(lines)
