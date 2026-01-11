 # ============================================
# XAI Stability & Consistency Experiment (FINAL – TREE SHAP, NO LEAKAGE)
# ============================================

import pickle
import pandas as pd
import numpy as np
import shap
from lime.lime_tabular import LimeTabularExplainer
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split

# --------------------------------------------
# Load model and data
# --------------------------------------------
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

df = pd.read_csv("Crop_recommendation.csv")

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
y = df["label"]

# --------------------------------------------
# Train–test split (FOR XAI BACKGROUND ONLY)
# --------------------------------------------
X_train, X_test = train_test_split(
    df[FEATURES],
    test_size=0.2,
    random_state=42
)

class_names = sorted(y.unique())

# --------------------------------------------
# Base input (anchor point)
# --------------------------------------------
base_input = pd.DataFrame([{
    "N": 90,
    "P": 42,
    "K": 43,
    "temperature": 20.8,
    "humidity": 82,
    "ph": 6.5,
    "rainfall": 202
}])

# --------------------------------------------
# SHAP TreeExplainer
# --------------------------------------------
shap_explainer = shap.TreeExplainer(model)

# --------------------------------------------
# LIME Explainer (TRAIN DATA ONLY)
# --------------------------------------------
lime_explainer = LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=FEATURES,
    class_names=class_names,
    mode="classification",
    random_state=42
)

# --------------------------------------------
# Experiment
# --------------------------------------------
NUM_RUNS = 5
shap_vectors = []
lime_feature_sets = []

for _ in range(NUM_RUNS):

    # small perturbation
    noise = np.random.normal(0, 0.5, base_input.shape)
    noisy_input = base_input + noise

    # -------- SHAP --------
    shap_values = shap_explainer.shap_values(noisy_input)

    # TreeExplainer multiclass-safe handling
    if isinstance(shap_values, list):
        shap_vals = shap_values[0][0]
    else:
        shap_vals = shap_values[0]

    shap_vals = shap_vals.flatten()
    shap_vals = shap_vals[:len(FEATURES)]

    shap_vectors.append(np.abs(shap_vals))

    # -------- LIME --------
    exp = lime_explainer.explain_instance(
        noisy_input.values[0],
        model.predict_proba,
        num_features=3
    )

    lime_set = set()
    for f, _ in exp.as_list():
        for feat in FEATURES:
            if feat in f:
                lime_set.add(FEATURES.index(feat))
                break

    lime_feature_sets.append(lime_set)

# --------------------------------------------
# SHAP Stability (Spearman)
# --------------------------------------------
if len(shap_vectors) > 1:
    spearman_scores = [
        spearmanr(shap_vectors[0], shap_vectors[i])[0]
        for i in range(1, len(shap_vectors))
    ]
    avg_spearman = np.mean(spearman_scores)
else:
    avg_spearman = 0.0

# --------------------------------------------
# LIME Consistency (Jaccard)
# --------------------------------------------
if len(lime_feature_sets) > 1:
    base_set = lime_feature_sets[0]
    jaccard_scores = [
        len(base_set & lime_feature_sets[i]) / len(base_set | lime_feature_sets[i])
        for i in range(1, len(lime_feature_sets))
    ]
    avg_jaccard = np.mean(jaccard_scores)
else:
    avg_jaccard = 0.0

# --------------------------------------------
# Results
# --------------------------------------------
print("====== XAI STABILITY RESULTS ======")
print(f"Average SHAP Spearman Correlation: {round(avg_spearman, 3)}")
print(f"Average LIME Jaccard Similarity:   {round(avg_jaccard, 3)}")
