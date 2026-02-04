import json
from pathlib import Path

import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt
from src.data.preprocess import clean_data, FEATURE_SET_STRICT, FEATURE_SET_FULL, TARGET
from src.data.split import split_data
from src.features.build_features import build_preprocessor


def get_feature_types(df, feature_set):
    numeric = [c for c in feature_set if df[c].dtype != "object"]
    categorical = [c for c in feature_set if df[c].dtype == "object"]
    return numeric, categorical


def train_and_eval_baseline_rf(df, feature_set, title="MODEL"):
    X = df[feature_set]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = split_data(X, y)

    num_cols, cat_cols = get_feature_types(df, feature_set)
    preprocessor = build_preprocessor(num_cols, cat_cols)

    rf = RandomForestRegressor(
        n_estimators=300,      # stable default-like
        random_state=42,
        n_jobs=-1
    )

    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", rf)
    ])

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, preds),
        "MSE": mean_squared_error(y_test, preds),
        "RMSE": sqrt(mean_squared_error(y_test, preds)),
        "R2": r2_score(y_test, preds)
    }

    return pipe, metrics


def save_artifacts(model, metadata, model_path: Path, meta_path: Path):
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    df = pd.read_csv("D:/ML_PROJECTS/co2-risk-platform/data/raw/co2.csv")
    df = clean_data(df)

    strict_model, strict_metrics = train_and_eval_baseline_rf(df, FEATURE_SET_STRICT, title="STRICT")
    full_model, full_metrics = train_and_eval_baseline_rf(df, FEATURE_SET_FULL, title="FULL")

    artifacts_dir = Path("artifacts/models")

    strict_meta = {
        "model_name": "RandomForestRegressor",
        "feature_set": "STRICT",
        "features": FEATURE_SET_STRICT,
        "target": TARGET,
        "metrics_holdout": strict_metrics,
        "notes": "Baseline RF selected via 5-fold CV; tuning did not improve.",
        "version": "v1"
    }

    full_meta = {
        "model_name": "RandomForestRegressor",
        "feature_set": "FULL",
        "features": FEATURE_SET_FULL,
        "target": TARGET,
        "metrics_holdout": full_metrics,
        "notes": "Baseline RF selected via 5-fold CV; tuning did not improve.",
        "version": "v1"
    }

    save_artifacts(
        strict_model,
        strict_meta,
        artifacts_dir / "rf_strict_v1.joblib",
        artifacts_dir / "rf_strict_v1.meta.json"
    )

    save_artifacts(
        full_model,
        full_meta,
        artifacts_dir / "rf_full_v1.joblib",
        artifacts_dir / "rf_full_v1.meta.json"
    )

    print("model Saved:")
    print(" - artifacts/models/rf_strict_v1.joblib")
    print(" - artifacts/models/rf_strict_v1.meta.json")
    print(" - artifacts/models/rf_full_v1.joblib")
    print(" - artifacts/models/rf_full_v1.meta.json")
    print("\nHoldout metrics:")
    print("STRICT:", strict_metrics)
    print("FULL:", full_metrics)
