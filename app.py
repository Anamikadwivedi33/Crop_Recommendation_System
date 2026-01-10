import matplotlib
matplotlib.use("Agg")   #  FINAL FIX

import pickle
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import time


from flask import Flask, render_template, request
from lime.lime_tabular import LimeTabularExplainer

from explain_utils import build_ideal_ranges, explain_crop, calculate_risk
from market_utils import get_mandi_data

app = Flask(__name__)

# -----------------------------
# Load trained model
# -----------------------------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# -----------------------------
# Load dataset (for SHAP & LIME background)
# -----------------------------
df = pd.read_csv("Crop_recommendation.csv")
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
X_df = df[FEATURES]
y = df["label"]
class_names = sorted(y.unique())

# -----------------------------
# Build ideal ranges once
# -----------------------------
ideal_ranges = build_ideal_ranges("Crop_recommendation.csv")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/market")
def market():
    data = get_mandi_data()
    return render_template("market.html", market_data=data)
@app.route("/predict", methods=["GET", "POST"])
def predict():
    results = None
    error = None

    if request.method == "POST":
        try:
            # -----------------------------
            # Read user inputs
            # -----------------------------
            user_input = {
                "N": float(request.form["N"]),
                "P": float(request.form["P"]),
                "K": float(request.form["K"]),
                "temperature": float(request.form["temperature"]),
                "humidity": float(request.form["humidity"]),
                "ph": float(request.form["ph"]),
                "rainfall": float(request.form["rainfall"])
            }

            input_df = pd.DataFrame([user_input])

            X = [[
                user_input["N"],
                user_input["P"],
                user_input["K"],
                user_input["temperature"],
                user_input["humidity"],
                user_input["ph"],
                user_input["rainfall"]
            ]]

            # -----------------------------
            # Model prediction
            # -----------------------------
            probs = model.predict_proba(X)[0]
            classes = model.classes_
            top3_idx = np.argsort(probs)[-3:][::-1]

            results = []
            for i in top3_idx:
                crop = classes[i].title()
                score = round(probs[i] * 100, 2)

                why, why_not, deviation = explain_crop(crop, user_input, ideal_ranges)
                risk = calculate_risk(deviation)

                results.append({
                    "crop": crop,
                    "suitability": f"{score}%",
                    "why": why,
                    "why_not": why_not,
                    "risk": risk
                })

            # -----------------------------
            # ✅ FIX 1: Risk-based sorting
            # -----------------------------
            risk_priority = {
                "Low Risk": 0,
                "Medium Risk": 1,
                "High Risk": 2
            }

            results = sorted(
                results,
                key=lambda r: (
                    risk_priority.get(r["risk"], 3),
                    -float(r["suitability"].replace("%", ""))
                )
            )

            # ======================================================
            # SHAP EXPLANATION
            # ======================================================
            background = X_df.sample(50, random_state=42)
            shap_explainer = shap.KernelExplainer(
                model.predict_proba,
                background
            )

            shap_values = shap_explainer.shap_values(input_df)

            plt.figure()
            shap.summary_plot(
                shap_values,
                input_df,
                plot_type="bar",
                show=False
            )
            plt.tight_layout()
            plt.savefig("static/explain/shap.png", bbox_inches="tight")
            plt.close()

            # ======================================================
            # LIME EXPLANATION
            # ======================================================
            lime_explainer = LimeTabularExplainer(
                training_data=X_df.values,
                feature_names=FEATURES,
                class_names=class_names,
                mode="classification"
            )

            exp = lime_explainer.explain_instance(
                input_df.values[0],
                model.predict_proba,
                num_features=7
            )

            fig = exp.as_pyplot_figure()
            plt.tight_layout()
            fig.savefig("static/explain/lime.png", bbox_inches="tight")
            plt.close(fig)

        except Exception as e:
            error = str(e)

    return render_template(
        "predict.html",
        results=results,
        error=error,
        ts=int(time.time())
    )
print("Starting Flask Server...")

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)

