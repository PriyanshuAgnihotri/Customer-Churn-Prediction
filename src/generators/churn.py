import numpy as np
import pandas as pd

from src.constants.churn_rules import (
    CHURN_BASE_PROBABILITY,
    DAYS_SINCE_LOGIN_THRESHOLD,
    DAYS_SINCE_LOGIN_WEIGHT,
    USAGE_CHANGE_THRESHOLD,
    USAGE_CHANGE_WEIGHT,
    SATISFACTION_THRESHOLD,
    SATISFACTION_WEIGHT,
    FAILED_PAYMENT_THRESHOLD,
    FAILED_PAYMENT_WEIGHT,
    SUPPORT_TICKET_THRESHOLD,
    SUPPORT_TICKET_WEIGHT,
    NON_RENEWAL_WEIGHT,
)


def generate_churn_signal(
    customers: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate churn probability and churn target."""

    risk = np.full(len(customers), CHURN_BASE_PROBABILITY)

    risk += np.where(
        customers["days_since_last_login"] >= DAYS_SINCE_LOGIN_THRESHOLD,
        DAYS_SINCE_LOGIN_WEIGHT,
        0,
    )

    risk += np.where(
        customers["usage_change_30d"] <= USAGE_CHANGE_THRESHOLD,
        USAGE_CHANGE_WEIGHT,
        0,
    )

    risk += np.where(
        customers["customer_satisfaction_score"] <= SATISFACTION_THRESHOLD,
        SATISFACTION_WEIGHT,
        0,
    )

    risk += np.where(
        customers["failed_payments_last_6m"] >= FAILED_PAYMENT_THRESHOLD,
        FAILED_PAYMENT_WEIGHT,
        0,
    )

    risk += np.where(
        customers["support_tickets_last_90d"] >= SUPPORT_TICKET_THRESHOLD,
        SUPPORT_TICKET_WEIGHT,
        0,
    )

    risk += np.where(
        customers["auto_renew"] == "No",
        NON_RENEWAL_WEIGHT,
        0,
    )

    risk = np.clip(risk, 0, 0.95)

    customers["churn_probability"] = risk
    customers["churn"] = rng.binomial(1, risk)

    return customers