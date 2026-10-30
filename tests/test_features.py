import pandas as pd

from src.features.build_features import build_features


def test_non_renewing_customer_flag():
    df = pd.DataFrame({
        "monthly_revenue": [250.0, 250.0],
        "monthly_logins": [10, 10],
        "support_tickets_last_90d": [2, 2],
        "active_days_last_30": [15, 15],
        "days_since_last_login": [5, 5],
        "usage_change_30d": [10.0, 10.0],
        "customer_satisfaction_score": [80, 80],
        "failed_payments_last_6m": [0, 0],
        "auto_renew": [0, 1],
    })

    result = build_features(df)

    assert result["is_non_renewing"].tolist() == [1, 0]