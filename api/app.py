from fastapi import FastAPI, Request
from fastapi.responses import Response
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from api.schemas import CustomerRequest, PredictionResponse
from src.models.predict import predict_churn


limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0.0",
)

app.state.limiter = limiter
app.add_exception_handler(
    RateLimitExceeded,
    _rate_limit_exceeded_handler,
)


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response: Response = await call_next(request)

    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cache-Control"] = "no-store"

    return response


@app.get("/health")
def health() -> dict:
    return {"status": "healthy"}


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
@limiter.limit("10/minute")
def predict(request: Request, customer: CustomerRequest):
    return predict_churn(customer.model_dump())
