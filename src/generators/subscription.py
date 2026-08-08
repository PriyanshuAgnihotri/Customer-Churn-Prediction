import numpy as np
import pandas as pd


from src.constants.subscription_rules import (
    PLAN_RULES,
    PLAN_PRICES,
    CONTRACT_RULES,
)


def generate_subscription_plan(
    customers: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate subscription plans."""

    plans = []

    for company_size in customers["company_size"]:

        probabilities = PLAN_RULES[company_size]

        plan = rng.choice(
            list(probabilities.keys()),
            p=list(probabilities.values()),
        )

        plans.append(plan)

    customers["subscription_plan"] = plans

    return customers


def generate_contract_type(
    customers: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate contract type."""

    contracts = []

    for plan in customers["subscription_plan"]:

        probabilities = CONTRACT_RULES[plan]

        contract = rng.choice(
            list(probabilities.keys()),
            p=list(probabilities.values()),
        )

        contracts.append(contract)

    customers["contract_type"] = contracts

    return customers


def generate_monthly_revenue(
    customers: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate monthly recurring revenue."""

    revenue = []

    for plan in customers["subscription_plan"]:

        base_price = PLAN_PRICES[plan]

        variation = rng.normal(
            loc=1.0,
            scale=0.05,
        )

        revenue.append(
            round(base_price * variation, 2)
        )

    customers["monthly_revenue"] = revenue

    return customers


def generate_auto_renew(
    customers: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate auto-renew flag."""

    customers["auto_renew"] = np.where(
        customers["contract_type"] == "Annual",
        rng.choice(
            [0, 1],
            size=len(customers),
            p=[0.10, 0.90],
        ),
        rng.choice(
            [0, 1],
            size=len(customers),
            p=[0.45, 0.55],
        ),
    )

    return customers
