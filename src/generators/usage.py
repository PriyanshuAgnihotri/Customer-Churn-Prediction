import numpy as np
import pandas as pd

from src.constants.usage_rules import (
    LOGIN_MEAN,
    ACTIVE_DAYS_MEAN,
    SESSION_DURATION,
    FEATURE_USAGE_MEAN,
)


def generate_monthly_logins(
    customers: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate monthly login counts."""

    customers["monthly_logins"] = customers["subscription_plan"].apply(
        lambda plan: max(
            0,
            int(rng.normal(LOGIN_MEAN[plan], 5)),
        )
    )

    return customers

def generate_active_days(
    customers: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate active days."""

    customers["active_days_last_30"] = customers["subscription_plan"].apply(
        lambda plan: min(
            30,
            max(
                0,
                int(rng.normal(ACTIVE_DAYS_MEAN[plan], 3)),
            ),
        )
    )

    return customers

def generate_session_duration(
    customers: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate average session duration."""

    customers["avg_session_minutes"] = customers["subscription_plan"].apply(
        lambda plan: round(
            max(
                2,
                rng.normal(SESSION_DURATION[plan], 4),
            ),
            1,
        )
    )

    return customers

def generate_feature_usage(
    customers: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate feature adoption score."""

    customers["feature_usage_score"] = customers["subscription_plan"].apply(
        lambda plan: round(
            min(
                100,
                max(
                    0,
                    rng.normal(FEATURE_USAGE_MEAN[plan], 8),
                ),
            ),
            1,
        )
    )

    return customers

def generate_last_login(
    customers: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate days since last login."""

    customers["days_since_last_login"] = (
        30 - customers["active_days_last_30"]
    ).clip(lower=0)

    return customers

def generate_usage_trend(
    customers: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate 30-day usage trend."""

    customers["usage_change_30d"] = rng.normal(
        loc=0,
        scale=20,
        size=len(customers),
    ).round(1)

    return customers

def generate_usage_metrics(
    customers: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate customer usage metrics."""

    customers = generate_monthly_logins(customers, rng)
    customers = generate_active_days(customers, rng)
    customers = generate_session_duration(customers, rng)
    customers = generate_feature_usage(customers, rng)
    customers = generate_last_login(customers, rng)
    customers = generate_usage_trend(customers, rng)

    return customers
