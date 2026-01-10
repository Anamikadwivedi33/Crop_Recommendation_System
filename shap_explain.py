 # shap_explain.py  (FINAL - FLASK SAFE)

import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os

# -----------------------------
# Load model & data
# -----------------------------
model = joblib.load("model.pkl")
df = pd.read_csv("Crop_recommendation.csv")

features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
X = df[features]

# -----------------------------
# Sample input (same as UI)
# -----------------------------
sample = X.iloc[[0]]   # ek hi row (IMPORTANT)

pred = model.predict(sample)[0]
print("Predicted Crop:", pred)

# -----------------------------
# SHAP Explainer
# -----------------------------
explainer = shap.TreeExplainer(model)
shap_values = explainer(sample)

# -----------------------------
# Create folder if not exists
# -----------------------------
os.makedirs("static/explain", exist_ok=True)

# -----------------------------
# GLOBAL SHAP BAR PLOT (SAFE)
# -----------------------------
plt.figure()
shap.summary_plot(
    shap_values.values,
    sample,
    plot_type="bar",
    show=False
)
plt.tight_layout()
plt.savefig("static/explain/shap.png")
plt.close()

print("SHAP image saved successfully")
