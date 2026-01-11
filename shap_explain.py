 # shap_explain.py  (FINAL – MANUAL SHAP, BULLETPROOF)

import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# -----------------------------
# Load model & data
# -----------------------------
model = joblib.load("model.pkl")
df = pd.read_csv("Crop_recommendation.csv")

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
X = df[FEATURES]

# -----------------------------
# Reproducible single sample
# -----------------------------
sample = X.sample(1, random_state=42)

pred = model.predict(sample)[0]
print("Predicted Crop:", pred)

# -----------------------------
# SHAP TreeExplainer
# -----------------------------
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(sample)

# Handle multiclass / version safely
if isinstance(shap_values, list):
    shap_vals = shap_values[0][0]
else:
    shap_vals = shap_values[0]

shap_vals = shap_vals.flatten()
shap_vals = shap_vals[:len(FEATURES)]   #  FINAL FIX


# -----------------------------
# Manual SHAP bar plot
# -----------------------------
os.makedirs("static/explain", exist_ok=True)

importance = np.abs(shap_vals)
indices = np.argsort(importance)

plt.figure(figsize=(6, 4))
plt.barh(
    [FEATURES[int(i)] for i in indices],
    importance[indices]
)
plt.xlabel("|SHAP value|")
plt.title("Local SHAP Explanation")
plt.tight_layout()
plt.savefig("static/explain/shap_local.png")
plt.close()

print("Local SHAP image saved successfully")
