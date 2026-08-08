import pandas as pd


def validate_dataset(df: pd.DataFrame) -> None:
    """Validate generated dataset."""

    required_columns = {
        "customer_id",
        "subscription_plan",
        "monthly_revenue",
        "monthly_logins",
        "active_days_last_30",
        "feature_usage_score",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}"
        )

    if df.empty:
        raise ValueError("Dataset is empty.")

    if not df["customer_id"].is_unique:
        raise ValueError("Duplicate customer IDs detected.")

    if df.isna().any().any():
        raise ValueError("Missing values detected.")

    if (df["monthly_revenue"] < 0).any():
        raise ValueError("Negative revenue detected.")

    if not df["active_days_last_30"].between(0, 30).all():
        raise ValueError("Invalid active-day values.")

    if not df["feature_usage_score"].between(0, 100).all():
        raise ValueError("Invalid feature usage scores.")