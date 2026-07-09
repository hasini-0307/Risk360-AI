# Risk360-AI
AI-powered multi-agent early warning system for loan default prediction using structured + unstructured data, explainable AI, retrieval, and intelligent recommendations.

# Risk360 AI Backend

## Modules

- `train.py` → Trains the CatBoost default prediction model.
- `predict.py` → Predicts loan default probability.
- `explain.py` → Generates SHAP feature explanations.
- `similarity.py` → Retrieves similar borrowers using ChromaDB and sentence embeddings.
- `predict_api.py` → FastAPI endpoints.

## Run the API

```bash
uvicorn backend.main:app --reload
```

## Swagger Documentation

Open:

http://127.0.0.1:8000/docs