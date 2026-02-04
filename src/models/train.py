import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from math import sqrt
from src.data.preprocess import(
    clean_data,
    TARGET,
    FEATURE_SET_FULL,
    FEATURE_SET_STRICT
)
from src.features.build_features import build_preprocessor
from src.data.split import split_data

def train_model(df, feature_set, model_name="baseline"):
    X = df[feature_set]
    y = df[TARGET]
    
    Xtrain, X_test, y_train, y_test = split_data(X, y)
    numeric_features = [col for col in feature_set if df[col].dtype != 'object']
    categorical_features = [col for col in feature_set if df[col].dtype == 'object']
    
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    
    model = LinearRegression()
    
    pip = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ]
    )
    
    pip.fit(Xtrain, y_train)
    
    y_pred = pip.predict(X_test)
    
    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "RMSE": sqrt(mean_squared_error(y_test, y_pred)),
        "R2": r2_score(y_test, y_pred)
    }
    
    return pip, metrics

if __name__ == "__main__":
    df = pd.read_csv("D:/ML_PROJECTS/co2-risk-platform/data/raw/co2.csv")
    df = clean_data(df)
    
    pipe_strict, metrics_strict = train_model(df, FEATURE_SET_STRICT, model_name="strict")
    pipe_full, metrics_full = train_model(df, FEATURE_SET_FULL, model_name="full")
    
    print("Strict Feature Set Metrics:")
    print(metrics_strict)
    
    print("Full Feature Set Metrics:")
    print(metrics_full)
    
    
    
                     