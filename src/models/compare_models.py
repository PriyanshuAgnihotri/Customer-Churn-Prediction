from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
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


def evaluate_model(name: str, model, X_test, y_test) -> dict:
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.50).astype(int)

    return {
        "model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }


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

    logistic_model = joblib.load(
        PROJECT_ROOT / "api" / "model.pkl"
    )

    random_forest_model = joblib.load(
        PROJECT_ROOT / "api" / "random_forest_model.pkl"
    )

    results = [
        evaluate_model(
            "Logistic Regression",
            logistic_model,
            X_test,
            y_test,
        ),
        evaluate_model(
            "Random Forest",
            random_forest_model,
            X_test,
            y_test,
        ),
    ]

    print(pd.DataFrame(results).round(3))


if __name__ == "__main__":
    main()