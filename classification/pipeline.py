"""Fit, evaluate, save, and load the cancellation classifier.

Classification only. The target is ``IsCanceled`` (binary 0/1). No regression model
lives in this package.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from .preprocessing import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    SOURCE_COLUMNS,
    TARGET,
    assert_no_leakage,
    build_preprocessor,
    load_classification_frame,
    prepare_inference_frame,
    sha256,
)

SEED = 42
MODEL_NAME = "logistic_classifier"
MODEL_DESCRIPTION = "scaled L2 logistic regression (binary classifier, balanced class weights)"
DEFAULT_ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"


def build_model(seed: int = SEED) -> LogisticRegression:
    """scikit-learn's LogisticRegression is a classifier, not a regressor."""
    return LogisticRegression(
        C=1.0,
        penalty="l2",
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced",
        random_state=seed,
    )


def build_pipeline(seed: int = SEED) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", build_preprocessor()),
            ("model", build_model(seed=seed)),
        ]
    )


def group_key(df: pd.DataFrame) -> np.ndarray:
    """Group exact-duplicate bookings so a duplicate never spans train/test.

    32.65% of H2 rows are duplicates. A random split would leak the same booking into
    both sides of the evaluation.
    """
    source_features = [c for c in FEATURES if c in SOURCE_COLUMNS]
    if not source_features:
        return np.arange(len(df), dtype=np.uint64)
    key = df[source_features].astype("string").fillna("<NA>").agg("|".join, axis=1)
    return pd.util.hash_pandas_object(key, index=False).to_numpy(dtype=np.uint64)


def split_frame(df: pd.DataFrame, seed: int = SEED) -> dict[str, np.ndarray]:
    """60/20/20 grouped split. Stratified where possible, grouped always."""
    y = df[TARGET].to_numpy(dtype=int)
    groups = group_key(df)
    X = np.zeros((len(df), 1))

    try:
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        folds = [test for _, test in sgkf.split(X, y, groups)]
        if len(folds) != 5:
            raise ValueError("unexpected fold count")
        out = {
            "train": np.concatenate(folds[2:]),
            "val": folds[1],
            "test": folds[0],
        }
    except Exception:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.40, random_state=seed)
        train_idx, temp_idx = next(gss.split(X, y, groups))
        gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
        val_rel, test_rel = next(gss2.split(X[temp_idx], y[temp_idx], groups[temp_idx]))
        out = {
            "train": np.sort(train_idx),
            "val": np.sort(temp_idx[val_rel]),
            "test": np.sort(temp_idx[test_rel]),
        }

    for name, idx in out.items():
        if len(idx) == 0:
            raise RuntimeError(f"empty {name} split")
    return out


def tune_threshold(y_true: np.ndarray, proba: np.ndarray) -> tuple[float, float]:
    """Pick the validation threshold that maximises F1; 0.5 is the fallback."""
    y_true = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype=float)
    thresholds = np.round(np.linspace(0.05, 0.95, 91), 4)
    best_t, best_score = 0.5, -1.0
    for t in thresholds:
        score = float(f1_score(y_true, (proba >= t).astype(int), zero_division=0))
        if score > best_score + 1e-12:
            best_t, best_score = float(t), score
        elif abs(score - best_score) <= 1e-12 and abs(float(t) - 0.5) < abs(best_t - 0.5):
            best_t = float(t)
    if best_score <= 0.0:
        return 0.5, 0.0
    return best_t, best_score


def classification_metrics(
    y_true: np.ndarray, proba: np.ndarray, threshold: float
) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype=float)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    out: dict[str, Any] = {
        "threshold": float(threshold),
        "n": int(len(y_true)),
        "positive_rate": float(y_true.mean()),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else None,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    if len(np.unique(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, proba))
        out["pr_auc"] = float(average_precision_score(y_true, proba))
    else:
        out["roc_auc"] = None
        out["pr_auc"] = None
    return out


def train_classification(
    data_path: str | Path,
    sample: int | None = None,
    seed: int = SEED,
    save: bool = True,
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Fit the preprocessing + classifier pipeline and optionally save the artifact."""
    src = Path(data_path)
    df = load_classification_frame(src)
    if sample is not None and sample < len(df):
        df = df.sample(n=int(sample), random_state=seed).reset_index(drop=True)

    assert_no_leakage(FEATURES)
    split = split_frame(df, seed=seed)
    y = df[TARGET].to_numpy(dtype=int)

    X_train = df.iloc[split["train"]][FEATURES]
    X_val = df.iloc[split["val"]][FEATURES]
    X_test = df.iloc[split["test"]][FEATURES]
    y_train = y[split["train"]]
    y_val = y[split["val"]]
    y_test = y[split["test"]]

    pipe = build_pipeline(seed=seed)
    pipe.fit(X_train, y_train)

    val_proba = pipe.predict_proba(X_val)[:, 1]
    threshold, val_best_f1 = tune_threshold(y_val, val_proba)
    test_proba = pipe.predict_proba(X_test)[:, 1]

    result: dict[str, Any] = {
        "task": "classification",
        "target": TARGET,
        "model": MODEL_NAME,
        "model_description": MODEL_DESCRIPTION,
        "seed": seed,
        "rows": int(len(df)),
        "n_features": int(len(FEATURES)),
        "threshold": float(threshold),
        "validation": {
            "at_0_5": classification_metrics(y_val, val_proba, 0.5),
            "at_tuned_threshold": classification_metrics(y_val, val_proba, threshold),
            "best_f1": float(val_best_f1),
        },
        "test": classification_metrics(y_test, test_proba, threshold),
        "feature_columns": FEATURES,
        "numeric_columns": NUMERIC_FEATURES,
        "categorical_columns": CATEGORICAL_FEATURES,
        "split_sizes": {k: int(len(v)) for k, v in split.items()},
        "data_path": str(src),
        "data_sha256": sha256(src),
        "sklearn_version": sklearn.__version__,
        "pipeline": pipe,
    }

    if save:
        artifact_path = save_artifact(result, out_dir=out_dir)
        result["artifact_path"] = str(artifact_path)

    return result


def save_artifact(result: dict[str, Any], out_dir: str | Path | None = None) -> Path:
    out = Path(out_dir) if out_dir is not None else DEFAULT_ARTIFACT_DIR
    out.mkdir(parents=True, exist_ok=True)
    artifact_path = out / f"{MODEL_NAME}.joblib"

    payload = {
        "task": "classification",
        "target": TARGET,
        "model_name": MODEL_NAME,
        "model_description": MODEL_DESCRIPTION,
        "pipeline": result["pipeline"],
        "threshold": float(result["threshold"]),
        "feature_columns": FEATURES,
        "numeric_columns": NUMERIC_FEATURES,
        "categorical_columns": CATEGORICAL_FEATURES,
        "seed": result["seed"],
        "sklearn_version": result["sklearn_version"],
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "data_sha256": result["data_sha256"],
        "metrics": {"validation": result["validation"], "test": result["test"]},
    }
    joblib.dump(payload, artifact_path, compress=3)

    digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    manifest = {
        "task": "classification",
        "target": TARGET,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "models": {
            MODEL_NAME: {
                "artifact": artifact_path.name,
                "sha256": digest,
                "algorithm": "LogisticRegression (binary classifier)",
                "threshold": float(result["threshold"]),
                "features": int(result["n_features"]),
                "rows": int(result["rows"]),
                "split_sizes": result["split_sizes"],
                "sklearn_version": result["sklearn_version"],
                "validation_f1_tuned": result["validation"]["at_tuned_threshold"]["f1"],
                "test_accuracy": result["test"]["accuracy"],
                "test_precision": result["test"]["precision"],
                "test_recall": result["test"]["recall"],
                "test_f1": result["test"]["f1"],
                "test_roc_auc": result["test"].get("roc_auc"),
                "test_pr_auc": result["test"].get("pr_auc"),
                "data_sha256": result["data_sha256"],
            }
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return artifact_path


def load_artifact(path: str | Path) -> dict[str, Any]:
    artifact = joblib.load(Path(path))
    if artifact.get("task") != "classification":
        raise ValueError(f"not a classification artifact: {path}")
    return artifact


def predict_cancel(
    artifact: dict[str, Any] | str | Path,
    rows: dict[str, Any] | pd.DataFrame,
    threshold: float | None = None,
) -> list[dict[str, Any]]:
    """Load a saved classifier and score bookings. Never retrains."""
    art = load_artifact(artifact) if isinstance(artifact, (str, Path)) else artifact
    if isinstance(rows, dict):
        frame = pd.DataFrame([rows])
    elif isinstance(rows, pd.Series):
        frame = rows.to_frame().T
    else:
        frame = pd.DataFrame(rows)

    X = prepare_inference_frame(frame)[art["feature_columns"]]
    assert_no_leakage(X.columns)
    proba = art["pipeline"].predict_proba(X)[:, 1]
    cut = float(art["threshold"] if threshold is None else threshold)
    return [
        {
            "cancel_probability": float(p),
            "is_canceled_pred": int(p >= cut),
            "threshold": cut,
        }
        for p in proba
    ]
