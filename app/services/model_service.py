from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "retail_demand_baseline.joblib"
)

FEATURES = [
    "lag_1",
    "lag_7",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28",
    "day_of_week",
    "month",
    "day_of_month",
]

_model = None


def load_model():
    global _model

    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found: {MODEL_PATH}"
            )

        _model = joblib.load(MODEL_PATH)

    return _model


def predict_demand(features: dict) -> float:

    missing = [
        feature
        for feature in FEATURES
        if feature not in features
    ]

    if missing:
        raise ValueError(
            f"Missing features: {missing}"
        )

    input_data = pd.DataFrame(
        [[features[name] for name in FEATURES]],
        columns=FEATURES,
    )

    model = load_model()

    prediction = float(model.predict(input_data)[0])

    return max(0.0, prediction)