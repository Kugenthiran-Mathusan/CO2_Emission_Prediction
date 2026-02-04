import pandas as pd
from pathlib import Path

def load_raw_data() -> pd.DataFrame:
    data_path = Path("D:/ML_PROJECTS/co2-risk-platform/data/raw/co2.csv")
    df = pd.read_csv(data_path)
    return df

if __name__ == "__main__":
    df = load_raw_data()
    print("shape: ", df.shape)
    print("columns: ", df.columns)
    print(df.head())