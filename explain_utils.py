import pandas as pd

# =====================================================
# Feature importance weights (domain-informed)
# =====================================================
FEATURE_WEIGHTS = {
    "rainfall": 2.0,
    "temperature": 2.0,
    "ph": 1.5,
    "humidity": 1.5
}

# =====================================================
# 1. Build ideal ranges from dataset
# =====================================================
def build_ideal_ranges(csv_path):
    df = pd.read_csv(csv_path)

    # clean labels
    df["label"] = df["label"].str.strip().str.title()

    features = ["temperature", "rainfall", "humidity", "ph"]
    ideal_ranges = {}

    for crop in df["label"].unique():
        crop_df = df[df["label"] == crop]
        ideal_ranges[crop] = {}

        for f in features:
            ideal_ranges[crop][f] = (
                crop_df[f].min(),
                crop_df[f].max()
            )

    return ideal_ranges


# =====================================================
# 2. Explain crop suitability (numerical + interpretable)
# =====================================================
def explain_crop(crop, user_input, ideal_ranges):
    why = []
    why_not = []
    deviation = {}   # feature-wise deviation magnitude

    if crop not in ideal_ranges:
        return ["Crop data not available"], ["Explanation not available"], {}

    for feature, (low, high) in ideal_ranges[crop].items():
        value = user_input[feature]

        if low <= value <= high:
            why.append(
                f"{feature} ({value}) lies within ideal range "
                f"({round(low,2)}–{round(high,2)})"
            )
            deviation[feature] = 0.0
        else:
            if value < low:
                diff = round(low - value, 2)
                why_not.append(
                    f"{feature} is {diff} units lower than ideal "
                    f"(ideal: {round(low,2)}–{round(high,2)})"
                )
            else:
                diff = round(value - high, 2)
                why_not.append(
                    f"{feature} is {diff} units higher than ideal "
                    f"(ideal: {round(low,2)}–{round(high,2)})"
                )

            deviation[feature] = diff

    return why, why_not, deviation


# =====================================================
# 3. Weighted risk calculation (robust & research-grade)
# =====================================================
def calculate_risk(deviation):
    total_risk = 0.0

    for feature, dev in deviation.items():
        weight = FEATURE_WEIGHTS.get(feature, 1.0)
        total_risk += float(dev) * weight

    if total_risk <= 3:
        return "Low Risk"
    elif total_risk <= 8:
        return "Medium Risk"
    else:
        return "High Risk"
