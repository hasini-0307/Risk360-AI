from fastapi import APIRouter

from backend.ml.predict import predict_default
from backend.ml.explain import explain_prediction
from backend.ml.similarity import search_similar_cases

from ai_engine.agents.behavior_agent import BehaviorAgent
from ai_engine.agents.income_agent import IncomeAgent
from ai_engine.agents.transaction_agent import TransactionAgent
from ai_engine.agents.document_agent import DocumentAgent

from ai_engine.cro.chief_risk_officer import ChiefRiskOfficer
from ai_engine.schemas.request_schema import AIRequest

router = APIRouter()


# -------------------------------------------------
# Predict only
# -------------------------------------------------
@router.post("/predict")
def predict(data: dict):
    return predict_default(data)


# -------------------------------------------------
# Explain only
# -------------------------------------------------
@router.post("/explain")
def explain(data: dict):

    explanation = explain_prediction(data)

    return {
        "top_features": explanation[:10]
    }


# -------------------------------------------------
# Similar Borrowers
# -------------------------------------------------
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
        similar_cases.append(
            {
                "LoanID": loan_id,
                "Distance": round(distance, 4)
            }
        )

    return {
        "similar_borrowers": similar_cases
    }


# -------------------------------------------------
# Complete Analysis
# -------------------------------------------------
@router.post("/analyze")
def analyze(data: dict):

    # ---------------------------------------------
    # ML Prediction
    # ---------------------------------------------
    prediction = predict_default(data)

    # ---------------------------------------------
    # SHAP Explanation
    # ---------------------------------------------
    explanation = explain_prediction(data)

    # ---------------------------------------------
    # Similar Borrowers
    # ---------------------------------------------
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
        similar_cases.append(
            {
                "LoanID": loan_id,
                "Distance": round(distance, 4)
            }
        )

    # ---------------------------------------------
    # Convert SHAP output into dictionary
    # ---------------------------------------------
    shap_dict = {
        feature: value
        for feature, value in explanation
    }

    # ---------------------------------------------
    # Build AI Request
    # ---------------------------------------------
    request = AIRequest(

        borrower_id=data.get("borrower_id", "UNKNOWN"),

        loan_id=data.get("loan_id", "UNKNOWN"),

        loan_type=data.get("loan_type", "Unknown"),

        default_probability=prediction["default_probability"],

        # Placeholder confidence
        model_confidence=0.90,

        borrower_profile=data,

        shap_values=shap_dict,

        branch_notes=data.get("branch_notes"),

        document_text=data.get("verification_notes"),

        similar_cases=similar_cases,
    )

    # ---------------------------------------------
    # Run Specialist Agents
    # ---------------------------------------------
    behavior = BehaviorAgent().analyze(request)

    income = IncomeAgent().analyze(request)

    transaction = TransactionAgent().analyze(request)

    document = DocumentAgent().analyze(request)

    # ---------------------------------------------
    # CRO Decision
    # ---------------------------------------------
    cro = ChiefRiskOfficer()

    cro_result = cro.evaluate(
        request,
        [
            behavior,
            income,
            transaction,
            document,
        ]
    )

    # ---------------------------------------------
    # Final Response
    # ---------------------------------------------
    return {

        "prediction": prediction["prediction"],

        "default_probability": prediction["default_probability"],

        "top_features": explanation[:10],

        "similar_borrowers": similar_cases,

        "cro_decision": cro_result.model_dump()
    }