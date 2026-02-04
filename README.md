
# 🚗 Vehicle CO2 Emission Prediction Platform (AI/ML Project)

Applink : https://vehicle-co2-emission-predictor-for-vehicle.streamlit.app/

Github : https://github.com/Kugenthiran-Mathusan/CO2_Emission_Prediction.git

Dataset : https://www.kaggle.com/datasets/brsahan/vehicle-co2-emissions-dataset

## 🔎 Overview
The **CO₂ emission for Vehicle Prediction Platform** is an end‑to‑end **Machine Learning system** designed to predict **vehicle CO₂ emissions (g/km)** and assess **emission risk** using a **custom, user‑defined threshold**.

This project is not a simple ML demo. It is designed to reflect **how a real AI/ML engineer works in industry**, covering **data understanding, model training, evaluation, explainability, system design, and deployment**.

The system allows users to:
- Predict CO₂ emissions for **new and unseen vehicles**
- Define their **own CO₂ risk limit**
- Get **risk decisions with explanations**
- Analyze **single vehicles or entire fleets**
- Interact with a **clean, production‑style web UI**

---

## 🎯 Problem Statement
Vehicle CO₂ emissions are a major contributor to climate change.  
While regulations such as EU policies focus on **fleet‑level averages**, many real‑world users (fleets, insurers, logistics companies, individuals) need **vehicle‑level decisions**.

### This system answers:
> “Given a vehicle’s specifications, how much CO₂ will it emit — and is that risky under my own chosen limit?”

---

## 🧠 What I Built (End‑to‑End)
This project demonstrates the **full lifecycle of an AI/ML system**:

1. Understanding the real‑world problem  
2. Exploring and preparing data  
3. Training multiple ML models  
4. Evaluating and selecting the best model  
5. Explaining model decisions  
6. Designing a realistic risk‑scoring system  
7. Building a user‑friendly interface  
8. Supporting batch (fleet) analysis  
9. Deploying the system as a live application  

---

## ⚙️ System Architecture

```
User (Streamlit Web UI)
        ↓
Validated Inputs (Select‑boxes & numeric ranges)
        ↓
Feature Encoding Pipeline
        ↓
Trained ML Model (Random Forest)
        ↓
CO₂ Prediction (g/km)
        ↓
Risk Scoring & Decision Logic
        ↓
PASS / AT_RISK / FAIL + Explanations
```

---

## 📊 Dataset
- Source: https://www.kaggle.com/datasets/brsahan/vehicle-co2-emissions-dataset 
- Contains:
  - Vehicle attributes (engine size, cylinders, transmission, fuel type, etc.)
  - Fuel consumption metrics
  - CO₂ emission values
- Used for:
  - Model training
  - Validation
  - Feature importance analysis

---

## 🤖 Machine Learning Details

### Models Trained & Compared
I trained and evaluated multiple models:
- Linear Regression
- Gradient Boosting Regressor
- **Random Forest Regressor (Final choice)**

### Why Random Forest?
- Excellent performance on tabular data
- Captures non‑linear relationships
- Robust to outliers and noise
- Provides feature importance (interpretability)

### Final Evaluation Results (Holdout Set)

| Mode   | MAE (g/km) | RMSE | R² Score |
|------|-----------|------|---------|
| STRICT | ~9.4 | ~13.4 | ~0.95 |
| FULL   | ~2.2 | ~4.0  | ~0.99 |

✔ FULL mode provides very high accuracy  
✔ STRICT mode works without fuel data (practical scenario)

---

## 🔍 Prediction Modes

### 1️⃣ STRICT Mode
- Uses **core vehicle attributes only**
- No fuel consumption required
- Useful when fuel data is unavailable
- Slightly lower accuracy, more practical

### 2️⃣ FULL Mode
- Uses **fuel consumption + engine data**
- Higher prediction accuracy
- Ideal for detailed assessments

---

## ⚠️ Risk Assessment System

Instead of hard‑coded legal rules, this system uses a **custom Vehicle Risk Limit**:

- User sets a CO₂ limit (g/km)
- System compares prediction against this limit
- Decision outcomes:

| Decision | Meaning |
|--------|--------|
| PASS | Vehicle safely below limit |
| AT_RISK | Close to the limit |
| FAIL | Exceeds the limit |

This makes the system flexible for **real‑world use**, not just policy simulation.

---

## 🧩 Explainability
Every prediction includes **human‑readable reasons**, such as:
- Large engine size increases CO₂
- High fuel consumption drives emissions
- Higher cylinder count increases fuel use

This ensures:
- Transparency
- Trust
- Non‑black‑box behavior

---

## 🖥️ User Interface Design

- Built with **Streamlit**
- Custom modern UI (not default Streamlit look)
- Select‑boxes only (no spelling mistakes)
- Friendly fuel types (users never see internal codes)
- Numeric inputs with validation & warnings

This UI design reflects **real industrial tools**.

---

## 📦 Fleet Batch Analysis

- Upload vehicle data as CSV
- Run predictions for entire fleet
- Get CO₂, risk score, and decision per vehicle
- Download enriched results

Used for:
- Fleet managers
- Emission audits
- Bulk analysis

---

## 🛠️ Tech Stack

### Machine Learning
- Python
- Pandas, NumPy
- scikit‑learn
- RandomForestRegressor
- joblib

### Application
- Streamlit
- Local ML inference (no API dependency)

### Deployment
- Streamlit Cloud
- Python 3.11

---

## 📁 Project Structure
```
co2‑risk‑platform/
│
├── app/
│   └── dashboard.py          # Streamlit UI
├── src/
│   ├── data/                 # Data utilities
│   ├── models/               # Training & evaluation
│   └── risk/                 # Risk scoring logic
├── artifacts/
│   └── models/               # Saved ML models
├── data/
│   └── raw/                  # Dataset
├── requirements.txt
├── runtime.txt
└── README.md
```

---

## 🚀 Live Deployment
This application is deployed on **Streamlit Cloud** and accessible via a public URL: https://vehicle-co2-emission-predictor-for-vehicle.streamlit.app/ 

---

## ▶️ Run Locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/dashboard.py
```

---

## 👨‍💻 About Me
**Mathusan**  
Aspiring **AI / Machine Learning Engineer** 

This project demonstrates my ability to:
- Think like a real ML engineer
- Build end‑to‑end ML systems
- Deliver explainable and deployable solutions

---
