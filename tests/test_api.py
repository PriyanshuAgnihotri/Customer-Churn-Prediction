from fastapi.testclient import TestClient

from api.app import app

client = TestClient(app)


VALID_CUSTOMER = {
    "tenure_months": 24,
    "country": "United States",
    "industry": "Technology",
    "company_size": "Medium",
    "subscription_plan": "Business",
    "contract_type": "Monthly",
    "monthly_revenue": 250.0,
    "auto_renew": "No",
    "monthly_logins": 12,
    "active_days_last_30": 7,
    "avg_session_minutes": 12.0,
    "feature_usage_score": 35.0,
    "days_since_last_login": 25,
    "usage_change_30d": -20.0,
    "support_tickets_last_90d": 4,
    "avg_resolution_hours": 30.0,
    "customer_satisfaction_score": 55.0,
    "failed_payments_last_6m": 2,
    "payment_method": "Credit Card",
}


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict_valid_customer():
    response = client.post("/predict", json=VALID_CUSTOMER)

    assert response.status_code == 200

    result = response.json()

    assert "churn_probability" in result
    assert "risk_level" in result
    assert isinstance(result["churn_probability"], float)
    assert isinstance(result["risk_level"], str)
    assert 0 <= result["churn_probability"] <= 1
    assert result["risk_level"] in {"Low", "Medium", "High"}


def test_predict_rejects_negative_tenure():
    customer = VALID_CUSTOMER.copy()
    customer["tenure_months"] = -1

    response = client.post("/predict", json=customer)

    assert response.status_code == 422


def test_predict_rejects_invalid_revenue():
    customer = VALID_CUSTOMER.copy()
    customer["monthly_revenue"] = -100

    response = client.post("/predict", json=customer)

    assert response.status_code == 422


def test_predict_rejects_invalid_feature_score():
    customer = VALID_CUSTOMER.copy()
    customer["feature_usage_score"] = 150

    response = client.post("/predict", json=customer)

    assert response.status_code == 422


def test_predict_rejects_invalid_active_days():
    customer = VALID_CUSTOMER.copy()
    customer["active_days_last_30"] = 31

    response = client.post("/predict", json=customer)

    assert response.status_code == 422


def test_predict_rejects_invalid_satisfaction_score():
    customer = VALID_CUSTOMER.copy()
    customer["customer_satisfaction_score"] = 101

    response = client.post("/predict", json=customer)

    assert response.status_code == 422


def test_predict_rejects_missing_required_field():
    customer = VALID_CUSTOMER.copy()
    del customer["country"]

    response = client.post("/predict", json=customer)

    assert response.status_code == 422