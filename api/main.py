from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from pathlib import Path
import pandas as pd
from src.risk.risk_scoring import risk_category_from_co2, risk_score_from_co2, generate_reasons
from src.risk.risk_scoring import fleet_compliance_summary


app = FastAPI(title="CO2 Risk & Compliance API", version="1.0")

# Load models once at startup (industrial practice)
ARTIFACTS = Path("artifacts/models")
STRICT_MODEL = joblib.load(ARTIFACTS / "rf_strict_v1.joblib")
FULL_MODEL = joblib.load(ARTIFACTS / "rf_full_v1.joblib")


class StrictInput(BaseModel):
    Make: str
    Vehicle_Class: str
    Transmission: str
    Fuel_Type: str
    Engine_Size_L: float
    Cylinders: int


class FullInput(StrictInput):
    Fuel_Consumption_Comb_L_100km: float


def to_strict_df(payload: StrictInput):
    return {
        "Make": payload.Make,
        "Vehicle Class": payload.Vehicle_Class,
        "Transmission": payload.Transmission,
        "Fuel Type": payload.Fuel_Type,
        "Engine Size(L)": payload.Engine_Size_L,
        "Cylinders": payload.Cylinders
    }


def to_full_df(payload: FullInput):
    row = to_strict_df(payload)
    row["Fuel Consumption Comb (L/100 km)"] = payload.Fuel_Consumption_Comb_L_100km
    return row


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict/strict")
def predict_strict(payload: StrictInput, limit: float = 200.0):
    row = to_strict_df(payload)

    X = pd.DataFrame([row])     # 1-row table
    co2_pred = float(STRICT_MODEL.predict(X)[0])


    return {
        "model": "rf_strict_v1",
        "co2_pred_g_km": round(co2_pred, 2),
        "risk_score": risk_score_from_co2(co2_pred, limit),
        "compliance": risk_category_from_co2(co2_pred, limit),
        "reasons": generate_reasons(row, mode="STRICT"),
        "limit_g_km": limit
    }


@app.post("/predict/full")
def predict_full(payload: FullInput, limit: float = 200.0):
    row = to_full_df(payload)
    X = pd.DataFrame([row])   # 1-row DataFrame (2D)
    co2_pred = float(FULL_MODEL.predict(X)[0])

    return {
        "model": "rf_full_v1",
        "co2_pred_g_km": round(co2_pred, 2),
        "risk_score": risk_score_from_co2(co2_pred, limit),
        "compliance": risk_category_from_co2(co2_pred, limit),
        "reasons": generate_reasons(row, mode="FULL"),
        "limit_g_km": limit
    }

from typing import List

class FleetCO2Input(BaseModel):
    co2_predictions: List[float]
    policy: str  # e.g. EU_2020_2024


@app.post("/fleet/compliance")
def fleet_compliance(payload: FleetCO2Input):
    return fleet_compliance_summary(
        co2_values=payload.co2_predictions,
        policy_key=payload.policy
    )
