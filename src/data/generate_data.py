"""Synthetic Customer Churn Dataset Generator."""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config() -> dict:
    """Load project configuration."""

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


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
        [
            "United States",
            "Canada",
            "United Kingdom",
            "Germany",
            "Australia",
            "India",
        ],
        size=n_customers,
        p=[0.40, 0.12, 0.15, 0.10, 0.08, 0.15],
    )

    customers["industry"] = rng.choice(
        [
            "Technology",
            "Finance",
            "Healthcare",
            "Retail",
            "Manufacturing",
            "Professional Services",
        ],
        size=n_customers,
        p=[0.25, 0.15, 0.15, 0.15, 0.12, 0.18],
    )

    customers["company_size"] = rng.choice(
        [
            "Small",
            "Mid-Market",
            "Enterprise",
        ],
        size=n_customers,
        p=[0.55, 0.30, 0.15],
    )

    return customers


def main() -> None:
    """Run data generation."""

    config = load_config()

    rng = create_rng(
        config["project"]["random_seed"]
    )

    customers = generate_customer_base(
    config,
    rng,
    )
    customers = generate_customer_profile(
    customers,
    rng,
    )

    print(customers.head())
    print("\nShape:", customers.shape)
    print("\nCompany Size")
    print(
    customers["company_size"]
    .value_counts(normalize=True)
    .round(3)
    )


if __name__ == "__main__":
    main()