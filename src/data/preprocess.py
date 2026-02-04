import pandas as pd

TARGET = "CO2 Emissions(g/km)"

NUMERIC_COLS = [
    "Engine Size(L)",
    "Cylinders",
    "Fuel Consumption Comb(L/100km)",
]

CATEGORICAL_COLS = [
    "Make",
    "Model",
    "Transmission",
    "Fuel Type"
]

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop_duplicates()
    df = df.dropna(subset=[TARGET])
    
    return df

FEATURE_SET_STRICT = [
    "Engine Size(L)",
    "Cylinders",
    "Make",
    "Vehicle Class",
    "Transmission",
    "Fuel Type"
]

FEATURE_SET_FULL = FEATURE_SET_STRICT + [
    "Fuel Consumption Comb (L/100 km)"
]