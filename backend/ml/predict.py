import pickle

import pandas as pd
from catboost import CatBoostClassifier


# -----------------------------
# Load trained model
# -----------------------------
model = CatBoostClassifier()
model.load_model("backend/models/catboost_model.cbm")

# -----------------------------
# Load feature order
# -----------------------------
with open("backend/models/feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)


# -----------------------------
# Predict
# -----------------------------
def predict_default(input_data):
    """
    input_data:
        dictionary containing borrower features

    Returns:
        prediction (0 or 1)
        probability of default
    """

    # Keep only the features the model was trained on
    model_input = {
        feature: input_data.get(feature)
        for feature in feature_columns
    }

    df = pd.DataFrame([model_input])

    # Ensure same feature order
    df = df[feature_columns]

    probability = model.predict_proba(df)[0][1]
    prediction = int(probability >= 0.5)

    return {
        "prediction": prediction,
        "default_probability": float(probability)
    }