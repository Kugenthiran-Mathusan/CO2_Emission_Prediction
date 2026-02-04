import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_validate

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from src.data.preprocess import clean_data, FEATURE_SET_STRICT, FEATURE_SET_FULL, TARGET
from src.features.build_features import build_preprocessor


def get_feature_types(df, feature_set):
    numeric_features = [c for c in feature_set if df[c].dtype != "object"]
    categorical_features = [c for c in feature_set if df[c].dtype == "object"]
    return numeric_features, categorical_features


def make_pipeline(preprocessor, model):
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])


def evaluate_models_cv(df, feature_set, title="STRICT"):
    X = df[feature_set]
    y = df[TARGET]

    numeric_features, categorical_features = get_feature_types(df, feature_set)
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingRegressor(random_state=42)
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # Use MAE as primary. sklearn gives NEGATIVE scores for losses.
    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2"
    }

    print(f"\n===== {title} FEATURE SET | 5-FOLD CV =====")

    results = []
    for name, model in models.items():
        pipe = make_pipeline(preprocessor, model)

        scores = cross_validate(
            pipe, X, y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False
        )

        mae = -scores["test_mae"]
        rmse = -scores["test_rmse"]
        r2 = scores["test_r2"]

        results.append({
            "model": name,
            "MAE_mean": mae.mean(),
            "MAE_std": mae.std(),
            "RMSE_mean": rmse.mean(),
            "RMSE_std": rmse.std(),
            "R2_mean": r2.mean(),
            "R2_std": r2.std()
        })

    # Sort by MAE (lower better)
    results = sorted(results, key=lambda x: x["MAE_mean"])

    for r in results:
        print(
            f"{r['model']:<18} | "
            f"MAE {r['MAE_mean']:.3f} ± {r['MAE_std']:.3f} | "
            f"RMSE {r['RMSE_mean']:.3f} ± {r['RMSE_std']:.3f} | "
            f"R2 {r['R2_mean']:.4f} ± {r['R2_std']:.4f}"
        )

    best = results[0]
    print(f"\nBest for {title}: {best['model']} (lowest MAE)\n")

    return results


if __name__ == "__main__":
    df = pd.read_csv("D:/ML_PROJECTS/co2-risk-platform/data/raw/co2.csv")
    df = clean_data(df)

    evaluate_models_cv(df, FEATURE_SET_STRICT, title="STRICT")
    evaluate_models_cv(df, FEATURE_SET_FULL, title="FULL")
