def risk_category_from_co2(co2_g_km: float, limit: float = 200.0):
    """
    Compliance-style categories.
    - PASS: comfortably below limit
    - AT_RISK: close to limit
    - FAIL: above limit
    """
    margin = 0.10 * limit  # 10% buffer
    if co2_g_km <= (limit - margin):
        return "PASS"
    elif co2_g_km <= limit:
        return "AT_RISK"
    else:
        return "FAIL"


def risk_score_from_co2(co2_g_km: float, limit: float = 200.0):
    """
    0–100 score. 50 ~ around limit.
    """
    # simple linear scaling with clipping
    score = (co2_g_km / limit) * 50
    return max(0, min(100, round(score, 1)))


def generate_reasons(input_row: dict, mode: str = "STRICT"):
    """
    Human-friendly reasons based on dominant features.
    mode: STRICT or FULL
    """
    reasons = []

    eng = input_row.get("Engine Size(L)")
    cyl = input_row.get("Cylinders")
    vclass = input_row.get("Vehicle Class")
    fuel_type = input_row.get("Fuel Type")
    comb = input_row.get("Fuel Consumption Comb (L/100 km)")

    # Common reasons
    if eng is not None and eng >= 3.0:
        reasons.append("Large engine size increases CO₂ emissions.")
    if cyl is not None and cyl >= 6:
        reasons.append("Higher cylinder count usually increases fuel use and CO₂.")
    if vclass is not None and ("SUV" in str(vclass) or "VAN" in str(vclass) or "PICKUP" in str(vclass)):
        reasons.append("Vehicle class (SUV/Van/Pickup) tends to have higher emissions.")

    # FULL mode: fuel consumption dominates
    if mode.upper() == "FULL":
        if comb is not None and comb >= 9.0:
            reasons.insert(0, "High combined fuel consumption is the main driver of CO₂ emissions.")
        if fuel_type in ["D", "E"]:
            reasons.append("Fuel type affects CO₂ output (diesel/ethanol blends can differ).")

    # Keep 3 reasons max
    if not reasons:
        reasons.append("Emissions are mainly influenced by engine and efficiency-related factors.")
    return reasons[:3]

EU_TARGETS = {
    "EU_2020_2024": 95.0,
    "EU_2025_2029": 93.6,
    "EU_2030_2034": 49.5,
    "EU_2035+": 0.0
}

EU_PENALTY_PER_G = 95.0  # € per g/km per vehicle


def fleet_compliance_summary(co2_values, policy_key: str):
    """
    EU-style fleet average compliance
    """
    target = EU_TARGETS[policy_key]

    avg_co2 = float(sum(co2_values) / len(co2_values))
    excess = max(0.0, avg_co2 - target)

    compliant = excess == 0.0

    penalty = excess * EU_PENALTY_PER_G * len(co2_values)

    return {
        "policy": policy_key,
        "target_g_km": target,
        "fleet_avg_g_km": round(avg_co2, 2),
        "compliant": compliant,
        "excess_g_km": round(excess, 2),
        "estimated_penalty_eur": round(penalty, 2)
    }
