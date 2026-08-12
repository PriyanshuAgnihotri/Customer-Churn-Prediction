import numpy as np
import pandas as pd

from src.constants.customer_rules import (
    COUNTRIES,
    COUNTRY_PROBABILITIES,
    INDUSTRIES,
    INDUSTRY_PROBABILITIES,
    COMPANY_SIZES,
    COMPANY_SIZE_PROBABILITIES,
)


def create_rng(seed: int) -> np.random.Generator:
    """Create reproducible random number generator."""

    return np.random.default_rng(seed)


def generate_customer_ids(
    n_customers: int,
) -> list[str]:
    """Generate customer IDs."""

    return [
        f"CUST_{customer_id:06d}"
        for customer_id in range(1, n_customers + 1)
    ]


def generate_customer_base(
    config: dict,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate customer lifecycle."""

    n_customers = config["data"]["n_customers"]

    snapshot_date = pd.Timestamp(
        config["data"]["snapshot_date"]
    )

    signup_days = rng.integers(
        low=30,
        high=365 * 5,
        size=n_customers,
    )

    signup_dates = (
        snapshot_date
        - pd.to_timedelta(signup_days, unit="D")
    )

    tenure_months = (
        signup_days / 30.44
    ).astype(int)

    return pd.DataFrame(
        {
            "customer_id": generate_customer_ids(n_customers),
            "signup_date": signup_dates,
            "snapshot_date": snapshot_date,
            "tenure_months": tenure_months,
        }
    )


def generate_customer_profile(
    customers: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate customer profile."""

    n_customers = len(customers)

    customers["country"] = rng.choice(
        COUNTRIES,
        size=n_customers,
        p=COUNTRY_PROBABILITIES,
    )

    customers["industry"] = rng.choice(
        INDUSTRIES,
        size=n_customers,
        p=INDUSTRY_PROBABILITIES,
    )

    customers["company_size"] = rng.choice(
        COMPANY_SIZES,
        size=n_customers,
        p=COMPANY_SIZE_PROBABILITIES,
    )

    return customers
