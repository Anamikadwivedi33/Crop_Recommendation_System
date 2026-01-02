import pickle
import numpy as np
from flask import Flask, render_template, request

# import json  <-- Removed as we use market_utils no
from explain_utils import build_ideal_ranges, explain_crop, calculate_risk
from market_utils import get_mandi_data

app = Flask(__name__)

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Mandi Data is now fetched dynamically inside the route

# Build ideal ranges once (dataset-derived)
# Build ideal ranges once (dataset-derived)
ideal_ranges = build_ideal_ranges("Crop_recommendation.csv")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/market")
def market():
    data = get_mandi_data()  # Fetches real or fallback data dynamically
    return render_template("market.html", market_data=data)

@app.route("/predict", methods=["GET", "POST"])
def predict():
    results = None
    error = None

    if request.method == "POST":
        try:
            # Read user inputs
            user_input = {
                "N": float(request.form["N"]),
                "P": float(request.form["P"]),
                "K": float(request.form["K"]),
                "temperature": float(request.form["temperature"]),
                "humidity": float(request.form["humidity"]),
                "ph": float(request.form["ph"]),
                "rainfall": float(request.form["rainfall"])
            }

            # Prepare model input (order must match training)
            X = [[
                user_input["N"],
                user_input["P"],
                user_input["K"],
                user_input["temperature"],
                user_input["humidity"],
                user_input["ph"],
                user_input["rainfall"]
            ]]

            # Predict probabilities
            probs = model.predict_proba(X)[0]
            classes = model.classes_

            # Pick TOP-3
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

        except Exception as e:
            error = str(e)

    return render_template("predict.html", results=results, error=error)

if __name__ == "__main__":
    app.run(debug=True)
