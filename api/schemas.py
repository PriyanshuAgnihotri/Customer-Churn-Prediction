from pydantic import BaseModel, Field


class CustomerRequest(BaseModel):
    tenure_months: int = Field(ge=0)
    country: str
    industry: str
    company_size: str
    subscription_plan: str
    contract_type: str
    monthly_revenue: float = Field(gt=0)
    auto_renew: str
    monthly_logins: int = Field(ge=0)
    active_days_last_30: int = Field(ge=0, le=30)
    avg_session_minutes: float = Field(ge=0)
    feature_usage_score: float = Field(ge=0, le=100)
    days_since_last_login: int = Field(ge=0)
    usage_change_30d: float
    support_tickets_last_90d: int = Field(ge=0)
    avg_resolution_hours: float = Field(ge=0)
    customer_satisfaction_score: float = Field(ge=0, le=100)
    failed_payments_last_6m: int = Field(ge=0)
    payment_method: str