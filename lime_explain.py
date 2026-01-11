 # lime_explain.py  (FINAL – RESEARCH SAFE)

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os
from lime.lime_tabular import LimeTabularExplainer

# -----------------------------
# Load model & data
# -----------------------------
model = joblib.load("model.pkl")
df = pd.read_csv("Crop_recommendation.csv")

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
X = df[FEATURES]
y = df["label"]

# -----------------------------
# Reproducible single sample
# -----------------------------
sample = X.sample(1, random_state=42).values[0]

pred = model.predict([sample])[0]
print("Predicted Crop:", pred)

# -----------------------------
# LIME Explainer (CONTROLLED)
# -----------------------------
explainer = LimeTabularExplainer(
    training_data=X.values,
    feature_names=FEATURES,
    class_names=sorted(y.unique()),
    mode="classification",
    random_state=42
)

# -----------------------------
# LIME Explanation (SPARSE)
# -----------------------------
exp = explainer.explain_instance(
    sample,
    model.predict_proba,
    num_features=3   # 🔥 sparsity for research
)

# -----------------------------
# Save explanation
# -----------------------------
os.makedirs("static/explain", exist_ok=True)

fig = exp.as_pyplot_figure()
plt.title("Local LIME Explanation")
plt.tight_layout()
fig.savefig("static/explain/lime_local.png", bbox_inches="tight")
plt.close(fig)

print("Local LIME image saved successfully")
