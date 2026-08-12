from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    model = joblib.load(
        PROJECT_ROOT / "api" / "random_forest_model.pkl"
    )

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    feature_names = preprocessor.get_feature_names_out()
    importance = classifier.feature_importances_

    result = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance": importance,
            }
        )
        .sort_values("importance", ascending=False)
        .head(20)
    )

    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
    