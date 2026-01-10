 # lime_explain.py  (FINAL - FLASK SAFE)

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

features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
X = df[features]
y = df["label"]

# -----------------------------
# Sample input
# -----------------------------
sample = X.iloc[0].values

pred = model.predict([sample])[0]
print("Predicted Crop:", pred)

# -----------------------------
# LIME Explainer
# -----------------------------
explainer = LimeTabularExplainer(
    training_data=X.values,
    feature_names=features,
    class_names=sorted(y.unique()),
    mode="classification"
)

exp = explainer.explain_instance(
    sample,
    model.predict_proba,
    num_features=7
)

# -----------------------------
# Create folder if not exists
# -----------------------------
os.makedirs("static/explain", exist_ok=True)

# -----------------------------
# SAVE LIME FIGURE
# -----------------------------
fig = exp.as_pyplot_figure()
fig.savefig("static/explain/lime.png", bbox_inches="tight")
plt.close(fig)

print("LIME image saved successfully")
