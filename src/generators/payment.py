import numpy as np
import pandas as pd

from src.constants.payment_rules import (
    FAILED_PAYMENT_MEAN,
    PAYMENT_METHODS,
    PAYMENT_METHOD_PROBABILITIES,
)


def generate_payment_metrics(
    customers: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate payment attributes."""

    customers["failed_payments_last_6m"] = rng.poisson(
        FAILED_PAYMENT_MEAN,
        size=len(customers),
    )

    customers["payment_method"] = rng.choice(
        PAYMENT_METHODS,
        size=len(customers),
        p=PAYMENT_METHOD_PROBABILITIES,
    )

    return customers