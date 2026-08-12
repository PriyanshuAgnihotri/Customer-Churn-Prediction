"""Payment generation rules."""

PAYMENT_METHODS = [
    "Credit Card",
    "ACH",
    "Wire Transfer",
]

PAYMENT_METHOD_PROBABILITIES = [
    0.55,
    0.30,
    0.15,
]

FAILED_PAYMENT_MEAN = 0.35