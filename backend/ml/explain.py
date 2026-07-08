import pickle

import pandas as pd
import shap
from catboost import CatBoostClassifier


# -----------------------------
# Load trained model
# -----------------------------
model = CatBoostClassifier()
model.load_model("backend/models/catboost_model.cbm")

# -----------------------------
# Load feature names
# -----------------------------
with open("backend/models/feature_columns.pkl", "rb") as f:
    feature_columns = pickle.load(f)

# -----------------------------
# SHAP Explainer
# -----------------------------
explainer = shap.TreeExplainer(model)


def explain_prediction(input_data):
    """
    input_data : dictionary of borrower features

    Returns:
        List of (feature, SHAP value) sorted by importance
    """

    df = pd.DataFrame([input_data])

    # Ensure exact same feature order used during training
    df = df[feature_columns]

    # SHAP values
    shap_values = explainer.shap_values(df)

    # CatBoost binary classifier may return either:
    # 1. numpy array
    # 2. list of arrays
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    contributions = {
        feature: float(value)
        for feature, value in zip(feature_columns, shap_values[0])
    }

    sorted_features = sorted(
        contributions.items(),
        key=lambda x: abs(x[1]),
        reverse=True,
    )

    return sorted_features