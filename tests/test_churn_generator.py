import numpy as np
import pandas as pd

from src.generators.churn import generate_churn_signal
from src.constants.churn_rules import NON_RENEWAL_WEIGHT


def test_non_renewal_increases_churn_probability():
    base = {
        "days_since_last_login": 0,
        "usage_change_30d": 10,
        "customer_satisfaction_score": 90,
        "failed_payments_last_6m": 0,
        "support_tickets_last_90d": 0,
    }

    renewing = pd.DataFrame([{**base, "auto_renew": 1}])
    non_renewing = pd.DataFrame([{**base, "auto_renew": 0}])

    renewing_result = generate_churn_signal(
        renewing, np.random.default_rng(42)
    )
    non_renewing_result = generate_churn_signal(
        non_renewing, np.random.default_rng(42)
    )

    expected_difference = NON_RENEWAL_WEIGHT

    actual_difference = (
        non_renewing_result["churn_probability"].iloc[0]
        - renewing_result["churn_probability"].iloc[0]
    )

    assert np.isclose(actual_difference, expected_difference)