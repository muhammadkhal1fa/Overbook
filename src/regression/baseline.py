"""
Milestone 1: untuned regression baselines for ADR.

Run from the project root:
    python -m src.regression.baseline --data data/H2.csv

Protocol (same for every model so the comparison is fair):
  * clean -> 80/20 split (seed 42) -> the 20% TEST part is NOT touched here
  * 5-fold cross-validation on the 80% TRAIN part only
  * the whole preprocessing sits inside the Pipeline, so it is re-fitted on the
    training folds of every split (no leakage from the validation fold)
"""
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from src.data.cleaning import clean_common, clean_for_regression, load_h2
from src.preprocessing.regression_features import build_preprocessor

SEED = 42


def get_regression_data(path, seed=SEED):
    df = load_h2(path)
    df, rep1 = clean_common(df)
    df, rep2 = clean_for_regression(df)
    X, y = df.drop(columns=["ADR"]), df["ADR"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    return X_train, X_test, y_train, y_test, {**rep1, **rep2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/H2.csv")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    X_train, X_test, y_train, y_test, report = get_regression_data(args.data)
    print("Cleaning report:", report)
    print(f"train={len(X_train)}  test={len(X_test)} (test set untouched)\n")

    models = {
        "Mean predictor (reference)": DummyRegressor(strategy="mean"),
        "Ridge": Ridge(alpha=1.0, random_state=SEED),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1),
        "XGBoost": XGBRegressor(random_state=SEED, n_jobs=-1),
    }
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    scoring = {"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error", "r2": "r2"}

    rows = []
    for name, model in models.items():
        pipe = Pipeline([("prep", build_preprocessor()), ("model", model)])
        res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
        rows.append({
            "model": name,
            "cv_MAE": -res["test_mae"].mean(), "cv_RMSE": -res["test_rmse"].mean(),
            "cv_R2": res["test_r2"].mean(), "cv_R2_std": res["test_r2"].std(),
            "fit_time_s": res["fit_time"].mean(),
        })
        print(f"{name:28s} MAE={rows[-1]['cv_MAE']:.2f}  RMSE={rows[-1]['cv_RMSE']:.2f}  "
              f"R2={rows[-1]['cv_R2']:.3f}  ({rows[-1]['fit_time_s']:.1f}s/fold)")

    table = pd.DataFrame(rows).round(3)
    table.to_csv(os.path.join(args.out, "regression_baseline_cv.csv"), index=False)
    print("\nSaved", os.path.join(args.out, "regression_baseline_cv.csv"))


if __name__ == "__main__":
    main()
