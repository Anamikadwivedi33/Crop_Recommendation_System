import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Load dataset
df = pd.read_csv("Crop_recommendation.csv")

# 2. Features & Target
X = df[["N","P","K","temperature","humidity","ph","rainfall"]]
y = df["label"]

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Model
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

# 5. Train
model.fit(X_train, y_train)

# 6. Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# 7. Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved successfully.")
