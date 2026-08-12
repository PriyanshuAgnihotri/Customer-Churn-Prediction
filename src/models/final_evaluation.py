from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from src.constants.model_rules import CHURN_THRESHOLD


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DROP_COLUMNS = [
    "customer_id",
    "signup_date",
    "snapshot_date",
    "churn_probability",
    "churn",
]

THRESHOLD = 0.45


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
    predictions = (probabilities >= CHURN_THRESHOLD).astype(int)

    print("Threshold:", CHURN_THRESHOLD)
    print("ROC-AUC:", round(roc_auc_score(y_test, probabilities), 3))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))


if __name__ == "__main__":
    main()