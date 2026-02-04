from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

def build_preprocessor(numeric_features, categorical_features):
    numeric_transformer = Pipeline(
        steps=[
        ('scaler', StandardScaler())
    ]
        )

    categorical_transformer = Pipeline(
        steps=[
        ('onehot', OneHotEncoder(
            handle_unknown='ignore',
            sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    return preprocessor