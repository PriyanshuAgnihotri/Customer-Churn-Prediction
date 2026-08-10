from fastapi import FastAPI

from api.schemas import CustomerRequest
from src.models.predict import predict_churn


app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0.0",
)


@app.get("/health")
def health() -> dict:
    return {"status": "healthy"}


@app.post("/predict")
def predict(customer: CustomerRequest) -> dict:
    return predict_churn(customer.model_dump())
