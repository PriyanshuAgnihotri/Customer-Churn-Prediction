from pathlib import Path

import joblib
import pandas as pd

from src.constants.model_rules import CHURN_THRESHOLD
from src.features.build_features import build_features


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "api" / "random_forest_model.pkl"


model = joblib.load(MODEL_PATH)


def predict_churn(customer: dict) -> dict:

    # Convert incoming API request to DataFrame
    df = pd.DataFrame([customer])

    # Build the same derived features used during training
    df = build_features(df)

    # Remove columns that were not used for model training
    drop_columns = [
        "customer_id",
        "signup_date",
        "snapshot_date",
        "churn_probability",
        "churn",
    ]

    df = df.drop(
        columns=[column for column in drop_columns if column in df.columns]
    )

    # Predict using the complete trained pipeline.
    # The pipeline handles categorical values such as:
    # "Yes", "No", "United States", "Business", etc.
    probability = model.predict_proba(df)[0, 1]

    if probability >= CHURN_THRESHOLD:
        risk_level = "High"
    elif probability >= 0.25:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    return {
        "churn_probability": round(float(probability), 4),
        "risk_level": risk_level,
    }