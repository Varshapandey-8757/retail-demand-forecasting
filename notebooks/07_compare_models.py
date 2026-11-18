import json
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
REPORTS_DIR = BASE_DIR / "reports"
OUTPUT_FILE = REPORTS_DIR / "model_comparison_report.json"

RF_REPORT = REPORTS_DIR / "model_evaluation.json"
PROPHET_BASE_REPORT = REPORTS_DIR / "prophet_model_evaluation.json"
PROPHET_HOLIDAYS_REPORT = REPORTS_DIR / "prophet_holidays_evaluation.json"


def generate_comparison():
    print("=" * 60)
    print("EVALUATION COMPARISON: PROPHET VS RANDOM FOREST BASELINE")
    print("=" * 60)

    with open(RF_REPORT, "r", encoding="utf-8") as f:
        rf = json.load(f)

    with open(PROPHET_BASE_REPORT, "r", encoding="utf-8") as f:
        pb = json.load(f)

    with open(PROPHET_HOLIDAYS_REPORT, "r", encoding="utf-8") as f:
        ph = json.load(f)

    comparison = {
        "dataset": "retail_sales_daily_top20.csv",
        "validation_window": {
            "start": "2016-03-28",
            "end": "2016-04-24",
            "days": 28,
        },
        "models": {
            "RandomForestRegressor": {
                "type": "Supervised ML with Lag Features",
                "mae": round(rf["mae"], 3),
                "rmse": round(rf["rmse"], 3),
                "wape_percent": round(rf["wape_percent"], 2),
                "note": "Evaluated using historical lag features; requires lagged sales inputs.",
            },
            "Prophet_Baseline": {
                "type": "Pure Time Series (Yearly + Weekly Seasonality)",
                "mae": round(pb["mae"], 3),
                "rmse": round(pb["rmse"], 3),
                "wape_percent": round(pb["wape_percent"], 2),
                "note": "Independent time series forecasting without needing actual daily lag inputs.",
            },
            "Prophet_With_US_Holidays": {
                "type": "Pure Time Series (Seasonality + US Country Holidays)",
                "mae": round(ph["mae"], 3),
                "rmse": round(ph["rmse"], 3),
                "wape_percent": round(ph["wape_percent"], 2),
                "note": "Best pure forecasting model. Captures national sales spikes and holiday dips.",
            },
        },
        "key_takeaways": [
            "Prophet operates as a true forward-looking forecaster (no daily actual lag dependencies).",
            "Adding US country holidays reduced Prophet RMSE from 18.529 to 18.473 and WAPE to 24.42%.",
            "Prophet forecasts serve the 30-day forward demand directly into the inventory restock pipeline.",
        ],
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    print(f"\nSaved comparison report to: {OUTPUT_FILE}")
    print("\nSummary Metrics Table:")
    summary_df = pd.DataFrame(
        [
            {
                "Model": "Random Forest Baseline",
                "MAE": rf["mae"],
                "RMSE": rf["rmse"],
                "WAPE (%)": rf["wape_percent"],
            },
            {
                "Model": "Prophet Baseline",
                "MAE": pb["mae"],
                "RMSE": pb["rmse"],
                "WAPE (%)": pb["wape_percent"],
            },
            {
                "Model": "Prophet + US Holidays",
                "MAE": ph["mae"],
                "RMSE": ph["rmse"],
                "WAPE (%)": ph["wape_percent"],
            },
        ]
    )
    print(summary_df.round(3).to_string(index=False))


if __name__ == "__main__":
    generate_comparison()
