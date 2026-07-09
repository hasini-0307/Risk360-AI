from fastapi import APIRouter

from backend.ml.predict import predict_default
from backend.ml.explain import explain_prediction
from backend.ml.similarity import search_similar_cases

router = APIRouter()


# -----------------------------
# Predict only
# -----------------------------
@router.post("/predict")
def predict(data: dict):

    return predict_default(data)


# -----------------------------
# Explain only
# -----------------------------
@router.post("/explain")
def explain(data: dict):

    explanation = explain_prediction(data)

    return {
        "top_features": explanation[:10]
    }


# -----------------------------
# Similar borrowers only
# -----------------------------
@router.post("/similar")
def similar(data: dict):

    query_text = (
        "Branch Notes: "
        + data.get("branch_notes", "")
        + "\n\nCall Summary: "
        + data.get("call_summary", "")
        + "\n\nVerification Notes: "
        + data.get("verification_notes", "")
    )

    results = search_similar_cases(query_text)

    similar_cases = []

    for loan_id, distance in zip(
        results["ids"][0],
        results["distances"][0]
    ):
        similar_cases.append({
            "LoanID": loan_id,
            "Distance": round(distance, 4)
        })

    return {
        "similar_borrowers": similar_cases
    }


# -----------------------------
# Analyze everything
# -----------------------------
@router.post("/analyze")
def analyze(data: dict):

    prediction = predict_default(data)

    explanation = explain_prediction(data)

    query_text = (
        "Branch Notes: "
        + data.get("branch_notes", "")
        + "\n\nCall Summary: "
        + data.get("call_summary", "")
        + "\n\nVerification Notes: "
        + data.get("verification_notes", "")
    )

    similar = search_similar_cases(query_text)

    similar_cases = []

    for loan_id, distance in zip(
        similar["ids"][0],
        similar["distances"][0]
    ):
        similar_cases.append({
            "LoanID": loan_id,
            "Distance": round(distance, 4)
        })

    return {
        "prediction": prediction["prediction"],
        "default_probability": prediction["default_probability"],
        "top_features": explanation[:10],
        "similar_borrowers": similar_cases
    }