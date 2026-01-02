import pandas as pd

# =====================================================
# 1. Build ideal ranges from dataset
# =====================================================
def build_ideal_ranges(csv_path):
    df = pd.read_csv(csv_path)

    # Clean crop labels
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
# 2. Explain crop suitability (LOW / HIGH direction)
# =====================================================
def explain_crop(crop, user_input, ideal_ranges):
    why = []
    why_not = []
    deviation = 0

    if crop not in ideal_ranges:
        return ["Crop data not available"], ["No explanation found"], 3

    for feature, (low, high) in ideal_ranges[crop].items():
        value = user_input[feature]

        if low <= value <= high:
            why.append(f"{feature} is within ideal range")
        else:
            deviation += 1

            if value < low:
                why_not.append(
                    f"{feature} is lower than ideal range "
                    f"(ideal: {round(low,2)}–{round(high,2)})"
                )
            else:
                why_not.append(
                    f"{feature} is higher than ideal range "
                    f"(ideal: {round(low,2)}–{round(high,2)})"
                )

    return why, why_not, deviation


# =====================================================
# 3. Risk calculation
# =====================================================
def calculate_risk(deviation):
    if deviation == 0:
        return "Low Risk"
    elif deviation == 1:
        return "Medium Risk"
    else:
        return "High Risk"
