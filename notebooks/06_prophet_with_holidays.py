import json
from pathlib import Path
import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Paths
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed" / "retail_sales_daily_top20.csv"
REPORTS_DIR = BASE_DIR / "reports"
CONFIG_PATH = BASE_DIR / "configs" / "prophet_config.json"

# Load config
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

SPLIT_DATE = pd.to_datetime(config["validation_window"]["train_end_date"])
TEST_END = pd.to_datetime(config["validation_window"]["test_end_date"])

print("=" * 60)
print("PROPHET MODEL EXPERIMENT: ADDING US HOLIDAYS")
print("=" * 60)

df = pd.read_csv(DATA_PATH, parse_dates=["date"])
series_list = df["series_id"].unique()

all_predictions = []

for i, series_id in enumerate(series_list, 1):
    series_df = df[df["series_id"] == series_id].copy()

    prophet_df = series_df[["date", "sales"]].rename(
        columns={"date": "ds", "sales": "y"}
    )

    train_df = prophet_df[prophet_df["ds"] <= SPLIT_DATE]
    test_df = prophet_df[
        (prophet_df["ds"] > SPLIT_DATE) & (prophet_df["ds"] <= TEST_END)
    ]

    # Initialize Prophet with built-in US holiday calendar
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
    )
    # Add US nationwide holidays (Thanksgiving, Christmas, Easter, etc.)
    model.add_country_holidays(country_name="US")

    model.fit(train_df)

    # Forecast validation period
    future = model.make_future_dataframe(periods=28, freq="D")
    forecast = model.predict(future)

    # Filter validation dates
    test_forecast = forecast[
        (forecast["ds"] > SPLIT_DATE) & (forecast["ds"] <= TEST_END)
    ][["ds", "yhat"]].copy()

    test_forecast = test_forecast.rename(
        columns={"ds": "date", "yhat": "predicted_sales"}
    )
    test_forecast["series_id"] = series_id

    merged = test_forecast.merge(
        series_df[["date", "sales", "store_id", "item_id"]],
        on="date",
        how="left",
    )
    merged["predicted_sales"] = np.maximum(0, merged["predicted_sales"])
    merged["absolute_error"] = np.abs(
        merged["sales"] - merged["predicted_sales"]
    )

    all_predictions.append(merged)
    print(f"[{i}/{len(series_list)}] Fitted series with holidays: {series_id}")

results_df = pd.concat(all_predictions, ignore_index=True)

y_true = results_df["sales"]
y_pred = results_df["predicted_sales"]

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
total_sales = y_true.sum()
wape = (
    (results_df["absolute_error"].sum() / total_sales * 100)
    if total_sales > 0
    else 0
)

print("\n" + "=" * 60)
print("EVALUATION WITH US HOLIDAYS ADDED")
print("=" * 60)
print(f"MAE:  {mae:.3f}  (Baseline was 13.115)")
print(f"RMSE: {rmse:.3f}  (Baseline was 18.529)")
print(f"WAPE: {wape:.2f}% (Baseline was 24.50%)")

# Save comparison report
report = {
    "model": "Prophet with US Holidays",
    "dataset": "retail_sales_daily_top20.csv",
    "series_count": int(len(series_list)),
    "train_end_date": str(SPLIT_DATE.date()),
    "test_start_date": "2016-03-28",
    "test_end_date": str(TEST_END.date()),
    "mae": float(mae),
    "rmse": float(rmse),
    "wape_percent": float(wape),
    "improvements": {
        "baseline_rmse": 18.529,
        "new_rmse": float(rmse),
        "rmse_delta": float(rmse - 18.529),
    },
}

report_file = REPORTS_DIR / "prophet_holidays_evaluation.json"
with open(report_file, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print(f"\nSaved evaluation to: {report_file}")