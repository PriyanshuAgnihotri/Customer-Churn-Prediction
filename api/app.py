from fastapi import FastAPI

from api.schemas import CustomerRequest, PredictionResponse
from src.models.predict import predict_churn


app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0.0",
)


@app.get("/health")
def health() -> dict:
    return {"status": "healthy"}


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(customer: CustomerRequest):
    return predict_churn(customer.model_dump())
