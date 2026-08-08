"""Synthetic Customer Churn Dataset Generator."""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from src.generators.usage import generate_usage_metrics


from src.generators.customer import (
    create_rng,
    generate_customer_base,
    generate_customer_profile,
)

from src.generators.subscription import (
    generate_subscription_plan,
    generate_contract_type,
    generate_monthly_revenue,
    generate_auto_renew,
)


from src.constants.usage_rules import (
    LOGIN_MEAN,
    ACTIVE_DAYS_MEAN,
    SESSION_DURATION,
    FEATURE_USAGE_MEAN,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


PLAN_RULES = {
    "Small": {
        "Basic": 0.60,
        "Pro": 0.30,
        "Business": 0.09,
        "Enterprise": 0.01,
    },
    "Mid-Market": {
        "Basic": 0.10,
        "Pro": 0.35,
        "Business": 0.45,
        "Enterprise": 0.10,
    },
    "Enterprise": {
        "Basic": 0.01,
        "Pro": 0.09,
        "Business": 0.35,
        "Enterprise": 0.55,
    },
}


def load_config() -> dict:
    """Load project configuration."""

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


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
    
    customers = generate_subscription_plan(
    customers,
    rng,
    )
    
    customers = generate_contract_type(
    customers,
    rng,
    )

    customers = generate_monthly_revenue(
    customers,
    rng,
    )

    customers = generate_auto_renew(
    customers,
    rng,
    )
    
    customers = generate_usage_metrics(
    customers,
    rng,
    )
    
    print(customers.head())
    print("\nShape:", customers.shape)
    print("\nRevenue Summary")
    print(
        customers["monthly_revenue"]
        .describe()
        .round(2)
    )

    print("\nContract Distribution")

    print(
        customers["contract_type"]
        .value_counts(normalize=True)
        .round(3)
    )

    print("\nAverage Revenue by Plan")

    print(
        customers.groupby("subscription_plan")["monthly_revenue"]
        .mean()
        .round(2)
    )
    
    output_path = PROJECT_ROOT / "data" / "raw" / "customer_churn.csv"

    customers.to_csv(
        output_path,
        index=False,
    )

    print(f"\nDataset saved to: {output_path}")
    print(f"Shape: {customers.shape}")
    
    print(customers.head())

    print("\nShape:", customers.shape)

    print("\nRevenue Summary")
    print(
        customers["monthly_revenue"]
        .describe()
        .round(2)
    )

    print("\nContract Distribution")
    print(
        customers["contract_type"]
        .value_counts(normalize=True)
        .round(3)
    )

    print("\nAverage Revenue by Plan")
    print(
        customers.groupby("subscription_plan")["monthly_revenue"]
        .mean()
        .round(2)
    )
    
    print("\nUsage Summary")
    print(
        customers[
            [
                "monthly_logins",
                "active_days_last_30",
                "avg_session_minutes",
                "feature_usage_score",
            ]
        ]
        .describe()
        .round(2)
    )

    print("\nAverage Usage by Subscription Plan")
    print(
        customers.groupby("subscription_plan")[
            [
            "monthly_logins",
            "feature_usage_score",
            ]
        ]
        .mean()
        .round(1)
    )


if __name__ == "__main__":
    main()