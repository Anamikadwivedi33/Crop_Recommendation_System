from explain_utils import build_ideal_ranges, explain_crop, calculate_risk

# =====================================================
# 1. Ideal ranges load karo
# =====================================================
ideal_ranges = build_ideal_ranges("Crop_recommendation.csv")

# Debug: available crops (optional)
print("Available crops (sample):", list(ideal_ranges.keys())[:5])

# =====================================================
# 2. Dummy user input (example)
# =====================================================
user_input = {
    "temperature": 25,
    "rainfall": 200,
    "humidity": 80,
    "ph": 6.5
}

# =====================================================
# 3. Crop test (CASE-SAFE)
# =====================================================
crop_name = "Rice"   # .title() format

why, why_not, deviation = explain_crop(
    crop_name,
    user_input,
    ideal_ranges
)

risk = calculate_risk(deviation)

# =====================================================
# 4. Output
# =====================================================
print("\nCROP:", crop_name)

print("\nWHY:")
for r in why:
    print(" -", r)

print("\nWHY NOT:")
for r in why_not:
    print(" -", r)

print("\nRISK LEVEL:", risk)
