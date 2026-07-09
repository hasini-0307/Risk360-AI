from fastapi import APIRouter

from backend.ml.predict import predict_default
from backend.ml.explain import explain_prediction
from backend.ml.similarity import search_similar_cases

router = APIRouter()


@router.post("/analyze")
def analyze(data: dict):

    # -------------------------
    # Prediction
    # -------------------------
    prediction = predict_default(data)

    # -------------------------
    # SHAP Explanation
    # -------------------------
    explanation = explain_prediction(data)

    # -------------------------
    # Build text query
    # -------------------------
    query_text = (
        "Branch Notes: "
        + data.get("branch_notes", "")
        + "\n\nCall Summary: "
        + data.get("call_summary", "")
        + "\n\nVerification Notes: "
        + data.get("verification_notes", "")
    )

    # -------------------------
    # Similar borrowers
    # -------------------------
    similar = search_similar_cases(query_text)

    similar_ids = similar["ids"][0]

    # -------------------------
    # Return everything
    # -------------------------
    return {
        "prediction": prediction["prediction"],
        "default_probability": prediction["default_probability"],
        "top_features": explanation[:10],
        "similar_borrowers": similar_ids
    }