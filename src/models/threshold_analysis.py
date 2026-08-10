from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DROP_COLUMNS = [
    "customer_id",
    "signup_date",
    "snapshot_date",
    "churn_probability",
    "churn",
]


def main() -> None:
    df = pd.read_csv(
        PROJECT_ROOT / "data" / "processed" / "customer_churn_processed.csv"
    )

    X = df.drop(columns=DROP_COLUMNS)
    y = df["churn"]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = joblib.load(
        PROJECT_ROOT / "api" / "random_forest_model.pkl"
    )

    probabilities = model.predict_proba(X_test)[:, 1]

    results = []

    for threshold in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        predictions = (
            probabilities >= threshold
        ).astype(int)

        results.append(
            {
                "threshold": threshold,
                "precision": precision_score(
                    y_test,
                    predictions,
                    zero_division=0,
                ),
                "recall": recall_score(
                    y_test,
                    predictions,
                    zero_division=0,
                ),
                "f1": f1_score(
                    y_test,
                    predictions,
                    zero_division=0,
                ),
            }
        )

    print(
        pd.DataFrame(results).round(3)
    )


if __name__ == "__main__":
    main()