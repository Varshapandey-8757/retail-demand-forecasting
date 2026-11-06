import json
from pathlib import Path
import numpy as np
import pandas as pd
from prophet import Prophet

# Base project paths
BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = BASE_DIR / "configs" / "prophet_config.json"

# Load settings from config
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

DATA_PATH = BASE_DIR / config["data_path"]
OUTPUT_DIR = BASE_DIR / "reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_VERSION = config["model_version"]
HORIZON_DAYS = config["future_forecast"]["horizon_days"]


def generate_future_forecasts():
    """Fits Prophet on full history and forecasts the next 30 days for each series."""
    print("=" * 60)
    print("PROPHET PIPELINE: GENERATE 30-DAY FUTURE FORECASTS")
    print("=" * 60)
    print(f"Loading data from: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    series_list = df["series_id"].unique()

    print(f"Total series to forecast: {len(series_list)}")
    print(f"Forecast horizon: {HORIZON_DAYS} days")

    all_forecasts = []

    for i, series_id in enumerate(series_list, 1):
        series_df = df[df["series_id"] == series_id].sort_values("date")
        store_id = series_df["store_id"].iloc[0]
        item_id = series_df["item_id"].iloc[0]

        # Reshape to Prophet format: ds (date) and y (sales)
        prophet_df = series_df[["date", "sales"]].rename(
            columns={"date": "ds", "sales": "y"}
        )

        # Fit Prophet model
        model = Prophet(
            yearly_seasonality=config["hyperparameters"]["yearly_seasonality"],
            weekly_seasonality=config["hyperparameters"]["weekly_seasonality"],
            daily_seasonality=config["hyperparameters"]["daily_seasonality"],
        )
        model.fit(prophet_df)

        # Make future dataframe for 30 days
        future = model.make_future_dataframe(periods=HORIZON_DAYS, freq="D")
        forecast = model.predict(future)

        # Filter only the future unobserved days
        last_historical_date = prophet_df["ds"].max()
        future_forecast = forecast[forecast["ds"] > last_historical_date].copy()

        # Format to match app.schemas.forecast.ForecastRecord
        series_output = pd.DataFrame(
            {
                "store_id": store_id,
                "item_id": item_id,
                "forecast_date": future_forecast["ds"].dt.date,
                "predicted_units": np.maximum(
                    0, np.round(future_forecast["yhat"], 2)
                ),
                "model_version": MODEL_VERSION,
            }
        )

        all_forecasts.append(series_output)
        print(f"[{i}/{len(series_list)}] Forecasted: {store_id} | {item_id}")

    combined = pd.concat(all_forecasts, ignore_index=True)

    # Save output CSV
    output_file = OUTPUT_DIR / "prophet_future_30d_forecasts.csv"
    combined.to_csv(output_file, index=False)

    print("\n" + "=" * 60)
    print(f"SUCCESS: Generated {len(combined)} forecast records.")
    print(f"Export saved to: {output_file}")
    print("=" * 60)
    return combined


if __name__ == "__main__":
    generate_future_forecasts()