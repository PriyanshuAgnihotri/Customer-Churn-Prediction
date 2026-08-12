import pandas as pd

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build derived customer features."""

    df = df.copy()

    df["revenue_per_login"] = (
        df["monthly_revenue"] / df["monthly_logins"].clip(lower=1)
    )

    df["support_ticket_rate"] = df["support_tickets_last_90d"] / 3

    df["engagement_ratio"] = df["active_days_last_30"] / 30

    df["is_inactive"] = (
        df["days_since_last_login"] >= 14
    ).astype(int)

    df["is_declining_usage"] = (
        df["usage_change_30d"] < 0
    ).astype(int)

    df["is_low_satisfaction"] = (
        df["customer_satisfaction_score"] < 60
    ).astype(int)

    df["is_payment_risk"] = (
        df["failed_payments_last_6m"] > 0
    ).astype(int)

    df["is_non_renewing"] = (
        df["auto_renew"] == "No"
    ).astype(int)

    return df