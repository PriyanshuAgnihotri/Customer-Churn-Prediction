from pydantic import (
    BaseModel,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from src.constants.customer_rules import (
    COUNTRIES,
    INDUSTRIES,
    COMPANY_SIZES,
)
from src.constants.subscription_rules import PLAN_PRICES, CONTRACT_RULES
from src.constants.payment_rules import PAYMENT_METHODS

ALLOWED_CATEGORIES = {
    "country": set(COUNTRIES),
    "industry": set(INDUSTRIES),
    "company_size": set(COMPANY_SIZES),
    "subscription_plan": set(PLAN_PRICES),
    "contract_type": {
        contract
        for plan_rules in CONTRACT_RULES.values()
        for contract in plan_rules
    },
    "payment_method": set(PAYMENT_METHODS),
}


class CustomerRequest(BaseModel):
    tenure_months: int = Field(ge=0)
    country: str
    industry: str
    company_size: str
    subscription_plan: str
    contract_type: str
    monthly_revenue: float = Field(gt=0)
    auto_renew: int = Field(ge=0, le=1)
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

    @field_validator(
        "country",
        "industry",
        "company_size",
        "subscription_plan",
        "contract_type",
        "payment_method",
    )
    @classmethod
    def validate_categories(cls, value: str, info: ValidationInfo) -> str:
        allowed_values = ALLOWED_CATEGORIES[info.field_name]

        if value not in allowed_values:
            raise ValueError(
                f"Unsupported value for {info.field_name}"
            )

        return value
    
    @model_validator(mode="after")
    def validate_contract_for_plan(self):
        allowed_contracts = CONTRACT_RULES[self.subscription_plan]

        if self.contract_type not in allowed_contracts:
            raise ValueError(
                "Contract type is not supported for the selected subscription plan"
            )

        return self


class PredictionResponse(BaseModel):
    churn_probability: float = Field(ge=0, le=1)
    risk_level: str
