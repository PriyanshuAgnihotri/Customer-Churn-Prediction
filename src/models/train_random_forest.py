import joblib
import pandas as pd

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    df = pd.read_csv(
        PROJECT_ROOT / "data" / "processed" / "customer_churn_processed.csv"
    )

    target = "churn"

    drop_columns = [
        "customer_id",
        "signup_date",
        "snapshot_date",
        "churn_probability",
        target,
    ]

    X = df.drop(columns=drop_columns)
    y = df[target]

    categorical_features = X.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    if "auto_renew" not in categorical_features:
        categorical_features.append("auto_renew")

    numeric_features = [
        column
        for column in X.columns
        if column not in categorical_features
        ]

    preprocessor = ColumnTransformer(
        [
            ("numeric", "passthrough", numeric_features),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
        ]
    )

    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=10,
                    min_samples_leaf=5,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model.fit(X_train, y_train)

    joblib.dump(
        model,
        PROJECT_ROOT / "api" / "random_forest_model.pkl",
    )

    print("Random Forest trained successfully.")
    print("Training rows:", len(X_train))
    print("Test rows:", len(X_test))


if __name__ == "__main__":
    main()