# app/dashboard.py
import requests
import streamlit as st
import pandas as pd

API_BASE = "http://127.0.0.1:8000"

# ---------- Page config ----------
st.set_page_config(
    page_title="CO₂ Vehicle Risk Platform",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Custom CSS (modern UI; not Streamlit default) ----------
CUSTOM_CSS = """
<style>
.stApp {
  background: radial-gradient(circle at top left, rgba(20, 184, 166, 0.10), transparent 45%),
              radial-gradient(circle at bottom right, rgba(59, 130, 246, 0.10), transparent 45%),
              #0b1220;
  color: #e5e7eb;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

h1, h2, h3 { color: #f9fafb !important; }

section[data-testid="stSidebar"] {
  background: rgba(255,255,255,0.03);
  border-right: 1px solid rgba(255,255,255,0.08);
}

.card {
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.04);
  border-radius: 18px;
  padding: 18px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.25);
}

.kpi {
  border: 1px solid rgba(255,255,255,0.10);
  background: rgba(255,255,255,0.06);
  border-radius: 18px;
  padding: 16px 16px;
}
.kpi .label { font-size: 12px; opacity: 0.85; }
.kpi .value { font-size: 26px; font-weight: 800; margin-top: 6px; }

.badge {
  display: inline-block;
  padding: 6px 10px;
  border-radius: 999px;
  font-weight: 800;
  font-size: 12px;
}
.badge-pass { background: rgba(34,197,94,0.18); color: #86efac; border: 1px solid rgba(34,197,94,0.35); }
.badge-risk { background: rgba(245,158,11,0.18); color: #fcd34d; border: 1px solid rgba(245,158,11,0.35); }
.badge-fail { background: rgba(239,68,68,0.18); color: #fca5a5; border: 1px solid rgba(239,68,68,0.35); }

.stButton button {
  border-radius: 12px !important;
  border: 1px solid rgba(255,255,255,0.18) !important;
  background: linear-gradient(90deg, rgba(20,184,166,0.85), rgba(59,130,246,0.85)) !important;
  color: white !important;
  font-weight: 800 !important;
  padding: 10px 14px !important;
}

.stNumberInput input, .stSelectbox div[data-baseweb="select"] {
  border-radius: 12px !important;
  border: 1px solid rgba(255,255,255,0.14) !important;
  background: rgba(255,255,255,0.06) !important;
  color: #e5e7eb !important;
}

[data-testid="stDataFrame"] {
  border-radius: 14px;
  border: 1px solid rgba(255,255,255,0.10);
  overflow: hidden;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------- Helpers ----------
def badge_html(compliance: str) -> str:
    c = (compliance or "").upper()
    if c == "PASS":
        return '<span class="badge badge-pass">PASS</span>'
    if c == "AT_RISK":
        return '<span class="badge badge-risk">AT RISK</span>'
    return '<span class="badge badge-fail">FAIL</span>'


def call_predict_strict(payload: dict, limit: float):
    r = requests.post(f"{API_BASE}/predict/strict", params={"limit": limit}, json=payload, timeout=20)
    r.raise_for_status()
    return r.json()


def call_predict_full(payload: dict, limit: float):
    r = requests.post(f"{API_BASE}/predict/full", params={"limit": limit}, json=payload, timeout=20)
    r.raise_for_status()
    return r.json()


def app_header():
    st.markdown(
        """
        <div class="card">
          <h1 style="margin:0;">🚗 CO₂ Vehicle Risk Platform</h1>
          <p style="margin:8px 0 0 0; opacity:0.85;">
            Choose values from dropdowns (no spelling issues) + set your own CO₂ limit → predict emissions → get risk score, PASS/AT_RISK/FAIL, and clear reasons.
          </p>
        </div>
        """,
        unsafe_allow_html=True
    )


# ---------- Header ----------
app_header()
st.write("")

# ---------- Sidebar (only what you want) ----------
with st.sidebar:
    st.markdown("### ⚙️ Controls")

    model_mode = st.radio(
        "Prediction Model",
        ["STRICT (No fuel consumption)", "FULL (With fuel consumption)"]
    )

    vehicle_limit = st.number_input(
        "Vehicle Risk Limit (g/km)",
        min_value=80.0,
        max_value=400.0,
        value=250.0,
        step=10.0,
        help="Your custom limit. PASS/AT_RISK/FAIL is based on this."
    )

    st.caption("Tip: Raise limit for easier PASS. Lower limit for stricter control.")

# ---------- Tabs ----------
tab1, tab2 = st.tabs(["🔎 Vehicle Predictor", "📦 Fleet Batch Upload"])

# ---------- Options (dropdown choices; no spelling needed) ----------
MAKE_OPTIONS = [
    "TOYOTA", "HONDA", "NISSAN", "FORD", "BMW", "MERCEDES-BENZ", "AUDI",
    "VOLKSWAGEN", "HYUNDAI", "KIA", "MAZDA", "TESLA", "OTHER"
]

VEHICLE_CLASS_OPTIONS = [
    "COMPACT", "MID-SIZE", "FULL-SIZE",
    "SUV - SMALL", "SUV - STANDARD",
    "PICKUP TRUCK - SMALL", "PICKUP TRUCK - STANDARD",
    "VAN - PASSENGER"
]

TRANSMISSION_OPTIONS = [
    "A5", "A6", "A7", "A8",
    "AS5", "AS6", "AS7", "AS8",
    "M5", "M6", "AM7",
    "AV", "CVT"
]

# Friendly fuel labels -> model codes
FUEL_TYPE_UI_TO_CODE = {
    "Regular Gasoline": "X",
    "Premium Gasoline": "Z",
    "Diesel": "D",
    "Ethanol (E85)": "E",
    "Natural Gas": "N"
}


# ---------- TAB 1: Single prediction ----------
with tab1:
    left, right = st.columns([1.1, 0.9], gap="large")

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("## Vehicle Details")

        make = st.selectbox("Make", MAKE_OPTIONS)
        vehicle_class = st.selectbox("Vehicle Class", VEHICLE_CLASS_OPTIONS)
        transmission = st.selectbox("Transmission", TRANSMISSION_OPTIONS)

        fuel_type_ui = st.selectbox(
            "Fuel Type",
            list(FUEL_TYPE_UI_TO_CODE.keys()),
            help="Shown in simple terms. Internally mapped for the ML model."
        )
        fuel_type = FUEL_TYPE_UI_TO_CODE[fuel_type_ui]

        engine_size = st.number_input(
            "Engine Size (L)",
            min_value=0.6,
            max_value=8.0,
            value=1.5,
            step=0.1,
            help="Examples: 1.0, 1.5, 2.0, 3.5"
        )

        cylinders = st.selectbox(
            "Cylinders",
            [3, 4, 5, 6, 8, 10, 12],
            index=1,
            help="Common values: 3, 4, 6, 8"
        )

        fuel_comb = None
        if model_mode.startswith("FULL"):
            fuel_comb = st.number_input(
                "Fuel Consumption Comb (L/100 km)",
                min_value=2.0,
                max_value=30.0,
                value=7.5,
                step=0.1,
                help="Higher value = more CO₂"
            )

        # small sanity warnings (optional, but professional)
        if engine_size > 6.0:
            st.warning("⚠️ Engine size is very high. Please double-check.")
        if model_mode.startswith("FULL") and fuel_comb is not None and fuel_comb < 3.0:
            st.warning("⚠️ Fuel consumption looks extremely low. Please confirm.")

        predict_btn = st.button("Predict CO₂ & Risk")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("## Decision Output")

        if predict_btn:
            try:
                if model_mode.startswith("STRICT"):
                    payload = {
                        "Make": make,
                        "Vehicle_Class": vehicle_class,
                        "Transmission": transmission,
                        "Fuel_Type": fuel_type,
                        "Engine_Size_L": float(engine_size),
                        "Cylinders": int(cylinders)
                    }
                    res = call_predict_strict(payload, limit=vehicle_limit)
                else:
                    payload = {
                        "Make": make,
                        "Vehicle_Class": vehicle_class,
                        "Transmission": transmission,
                        "Fuel_Type": fuel_type,
                        "Engine_Size_L": float(engine_size),
                        "Cylinders": int(cylinders),
                        "Fuel_Consumption_Comb_L_100km": float(fuel_comb)
                    }
                    res = call_predict_full(payload, limit=vehicle_limit)

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(
                        f"""
                        <div class="kpi">
                          <div class="label">Predicted CO₂</div>
                          <div class="value">{res["co2_pred_g_km"]} <span style="font-size:14px;opacity:0.8;">g/km</span></div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                with c2:
                    st.markdown(
                        f"""
                        <div class="kpi">
                          <div class="label">Risk Score</div>
                          <div class="value">{res["risk_score"]} <span style="font-size:14px;opacity:0.8;">/ 100</span></div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                with c3:
                    st.markdown(
                        f"""
                        <div class="kpi">
                          <div class="label">Decision (Your Limit)</div>
                          <div class="value">{badge_html(res["compliance"])}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                st.write("")
                st.markdown("### 🔎 Top Reasons")
                for r in res["reasons"]:
                    st.markdown(f"- {r}")

                st.caption(f'Model used: **{res["model"]}** | Your Limit: **{res["limit_g_km"]} g/km**')

            except Exception as e:
                st.error(f"API error: {e}")
        else:
            st.info("Select details and click **Predict CO₂ & Risk**.")
        st.markdown("</div>", unsafe_allow_html=True)

# ---------- TAB 2: Batch upload ----------
with tab2:
    st.markdown(
        """
        <div class="card">
          <h2 style="margin:0;">Fleet Batch Upload</h2>
          <p style="margin:8px 0 0 0; opacity:0.85;">
            Upload a CSV → get CO₂ + risk decisions for all vehicles using your custom limit.
          </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.write("")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📄 Upload CSV")

    st.caption("Required columns (STRICT): Make, Vehicle_Class, Transmission, Fuel_Type, Engine_Size_L, Cylinders")
    st.caption("Extra column for FULL: Fuel_Consumption_Comb_L_100km")
    st.caption("Fuel_Type in CSV should be a code: X/Z/D/E/N (we can add friendly upload later).")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file is not None:
        df = pd.read_csv(file)
        st.markdown("### Preview")
        st.dataframe(df, use_container_width=True)

        run_btn = st.button("Run Batch Predictions")

        if run_btn:
            outputs = []
            for _, row in df.iterrows():
                try:
                    base = {
                        "Make": str(row["Make"]),
                        "Vehicle_Class": str(row["Vehicle_Class"]),
                        "Transmission": str(row["Transmission"]),
                        "Fuel_Type": str(row["Fuel_Type"]),  # expects code
                        "Engine_Size_L": float(row["Engine_Size_L"]),
                        "Cylinders": int(row["Cylinders"]),
                    }

                    if model_mode.startswith("FULL") and "Fuel_Consumption_Comb_L_100km" in df.columns:
                        base["Fuel_Consumption_Comb_L_100km"] = float(row["Fuel_Consumption_Comb_L_100km"])
                        res = call_predict_full(base, limit=vehicle_limit)
                    else:
                        res = call_predict_strict(base, limit=vehicle_limit)

                    outputs.append({
                        **base,
                        "co2_pred_g_km": res["co2_pred_g_km"],
                        "risk_score": res["risk_score"],
                        "decision": res["compliance"],
                        "model": res["model"],
                    })
                except Exception as e:
                    out = row.to_dict()
                    out["error"] = str(e)
                    outputs.append(out)

            out_df = pd.DataFrame(outputs)
            st.markdown("### ✅ Results")
            st.dataframe(out_df, use_container_width=True)

            st.download_button(
                label="Download Results CSV",
                data=out_df.to_csv(index=False).encode("utf-8"),
                file_name="fleet_co2_risk_results.csv",
                mime="text/csv"
            )

    st.markdown("</div>", unsafe_allow_html=True)
