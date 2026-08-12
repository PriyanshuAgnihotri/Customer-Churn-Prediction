import numpy as np
import pandas as pd

from src.constants.support_rules import (
    TICKET_RATE_BY_PLAN,
    RESOLUTION_HOURS_BY_PLAN,
)


def generate_support_tickets(
    customers: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate support ticket counts."""

    tickets = []

    for _, customer in customers.iterrows():
        base_rate = TICKET_RATE_BY_PLAN[customer["subscription_plan"]]

        usage_factor = max(
            0.5,
            1 + (50 - customer["feature_usage_score"]) / 100,
        )

        tickets.append(
            rng.poisson(base_rate * usage_factor)
        )

    customers["support_tickets_last_90d"] = tickets

    return customers


def generate_resolution_time(
    customers: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate average ticket resolution time."""

    customers["avg_resolution_hours"] = [
        round(
            max(
                2,
                rng.normal(
                    RESOLUTION_HOURS_BY_PLAN[plan],
                    5,
                ),
            ),
            1,
        )
        for plan in customers["subscription_plan"]
    ]

    return customers


def generate_satisfaction_score(
    customers: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate customer satisfaction score."""

    score = (
        70
        + customers["feature_usage_score"] * 0.15
        - customers["support_tickets_last_90d"] * 2
        - customers["avg_resolution_hours"] * 0.2
        + rng.normal(0, 5, len(customers))
    )

    customers["customer_satisfaction_score"] = (
        score.clip(0, 100).round(1)
    )

    return customers


def generate_support_metrics(
    customers: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate customer support metrics."""

    customers = generate_support_tickets(customers, rng)
    customers = generate_resolution_time(customers, rng)
    customers = generate_satisfaction_score(customers, rng)

    return customers
