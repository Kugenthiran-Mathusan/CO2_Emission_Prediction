import joblib
import pandas as pd
from pathlib import Path


def load_model(path):
    return joblib.load(path)


def get_feature_importance(model_pipeline, top_n=15):
    """
    Works for tree-based models (RandomForest).
    """
    preprocessor = model_pipeline.named_steps["preprocessor"]
    model = model_pipeline.named_steps["model"]

    # Get feature names after preprocessing
    feature_names = preprocessor.get_feature_names_out()

    importances = model.feature_importances_

    fi = (
        pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        })
        .sort_values("importance", ascending=False)
        .head(top_n)
    )

    return fi


if __name__ == "__main__":
    artifacts = Path("artifacts/models")

    strict_model = load_model(artifacts / "rf_strict_v1.joblib")
    full_model = load_model(artifacts / "rf_full_v1.joblib")

    strict_fi = get_feature_importance(strict_model)
    full_fi = get_feature_importance(full_model)

    print("\n=== STRICT MODEL FEATURE IMPORTANCE ===")
    print(strict_fi)

    print("\n=== FULL MODEL FEATURE IMPORTANCE ===")
    print(full_fi)
