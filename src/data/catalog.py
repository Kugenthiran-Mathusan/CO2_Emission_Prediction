import pandas as pd

CATALOG_COLS = [
    "Make",
    "Model",
    "Vehicle Class",
    "Transmission",
    "Fuel Type",
    "Engine Size(L)",
    "Cylinders",
    "Fuel Consumption Comb (L/100 km)",
]

def load_catalog(path="D:/ML_PROJECTS/co2-risk-platform/data/raw/co2.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    # keep only needed cols
    df = df[CATALOG_COLS].dropna()
    # remove duplicates (same spec repeated)
    df = df.drop_duplicates().reset_index(drop=True)
    return df
