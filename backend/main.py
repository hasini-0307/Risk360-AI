from fastapi import FastAPI

from backend.api.predict_api import router as predict_router

app = FastAPI(
    title="Risk360 AI API",
    version="1.0"
)

# Register API routes
app.include_router(predict_router)


@app.get("/")
def home():
    return {
        "message": "Risk360 AI Backend is running!"
    }