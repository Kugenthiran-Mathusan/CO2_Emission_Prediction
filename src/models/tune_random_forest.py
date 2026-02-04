import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestRegressor

from src.data.preprocess import (
    clean_data, FEATURE_SET_STRICT, FEATURE_SET_FULL, TARGET
)
from src.features.build_features import build_preprocessor


def get_feature_types(df, feature_set):
    numeric = [c for c in feature_set if df[c].dtype != "object"]
    categorical = [c for c in feature_set if df[c].dtype == "object"]
    return numeric, categorical


def tune_rf(df, feature_set, title="MODEL"):
    X = df[feature_set]
    y = df[TARGET]

    num_cols, cat_cols = get_feature_types(df, feature_set)
    preprocessor = build_preprocessor(num_cols, cat_cols)

    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(random_state=42))
    ])

    param_grid = {
        "model__n_estimators": [200, 300],
        "model__max_depth": [None, 15, 25],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [1, 2]
    }

    grid = GridSearchCV(
        pipe,
        param_grid,
        cv=5,
        scoring="neg_mean_absolute_error",
        n_jobs=-1
    )

    grid.fit(X, y)

    print(f"\n===== {title} RANDOM FOREST TUNING =====")
    print("Best MAE:", -grid.best_score_)
    print("Best Params:", grid.best_params_)

    return grid.best_estimator_


if __name__ == "__main__":
    df = pd.read_csv("D:/ML_PROJECTS/co2-risk-platform/data/raw/co2.csv")
    df = clean_data(df)

    best_rf_strict = tune_rf(df, FEATURE_SET_STRICT, title="STRICT")
    best_rf_full = tune_rf(df, FEATURE_SET_FULL, title="FULL")