from fastapi.testclient import TestClient

from api.app import app


client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    
    
def test_predict_valid_customer():
    customer = {
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

    response = client.post("/predict", json=customer)

    assert response.status_code == 200

    result = response.json()

    assert "churn_probability" in result
    assert "risk_level" in result
    assert isinstance(result["churn_probability"], float)
    assert isinstance(result["risk_level"], str)
    assert 0 <= result["churn_probability"] <= 1
    assert result["risk_level"] in {"Low", "Medium", "High"}
    
    
def test_predict_invalid_customer():
    customer = {
        "tenure_months": -1,
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

    response = client.post("/predict", json=customer)

    assert response.status_code == 422
    
    
def test_predict_rejects_invalid_revenue():
    response = client.post(
        "/predict",
        json={
            "tenure_months": 24,
            "country": "United States",
            "industry": "Technology",
            "company_size": "Medium",
            "subscription_plan": "Business",
            "contract_type": "Monthly",
            "monthly_revenue": -100,
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
            "revenue_per_login": 20.83,
            "support_ticket_rate": 0.13,
            "engagement_ratio": 0.23,
            "is_inactive": 0,
            "is_declining_usage": 1,
            "is_low_satisfaction": 1,
            "is_payment_risk": 1,
            "is_non_renewing": 1
        },
    )

    assert response.status_code == 422
    
    
def test_predict_rejects_invalid_feature_score():
    response = client.post(
        "/predict",
        json={
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
            "feature_usage_score": 150.0,
            "days_since_last_login": 25,
            "usage_change_30d": -20.0,
            "support_tickets_last_90d": 4,
            "avg_resolution_hours": 30.0,
            "customer_satisfaction_score": 55.0,
            "failed_payments_last_6m": 2,
            "payment_method": "Credit Card",
            "revenue_per_login": 20.83,
            "support_ticket_rate": 0.13,
            "engagement_ratio": 0.23,
            "is_inactive": 0,
            "is_declining_usage": 1,
            "is_low_satisfaction": 1,
            "is_payment_risk": 1,
            "is_non_renewing": 1
        },
    )

    assert response.status_code == 422
    
    