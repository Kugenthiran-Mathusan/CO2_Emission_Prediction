from src.risk.risk_scoring import risk_category_from_co2, risk_score_from_co2, generate_reasons

sample = {
    "Engine Size(L)": 3.5,
    "Cylinders": 6,
    "Vehicle Class": "SUV - STANDARD",
    "Fuel Type": "X",
    "Fuel Consumption Comb (L/100 km)": 10.2
}

co2_pred = 230.0

print("Category:", risk_category_from_co2(co2_pred, limit=200))
print("Score:", risk_score_from_co2(co2_pred, limit=200))
print("Reasons STRICT:", generate_reasons(sample, mode="STRICT"))
print("Reasons FULL:", generate_reasons(sample, mode="FULL"))
