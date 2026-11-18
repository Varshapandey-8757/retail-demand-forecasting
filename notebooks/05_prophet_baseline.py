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
MODELS_DIR = BASE_DIR / "models"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load data
print("Loading processed sales data...")
df = pd.read_csv(DATA_PATH, parse_dates=["date"])

# 2. Validation window (same 28 days as team baseline)
# Train end: 2016-03-27, Validation: 2016-03-28 to 2016-04-24
SPLIT_DATE = pd.to_datetime("2016-03-27")
TEST_END = pd.to_datetime("2016-04-24")

all_predictions = []
series_list = df["series_id"].unique()
print(f"Fitting Prophet models for {len(series_list)} series...")

for i, series_id in enumerate(series_list, 1):
    series_df = df[df["series_id"] == series_id].copy()

    # Prophet requires 'ds' (datestamp) and 'y' (target)
    prophet_df = series_df[["date", "sales"]].rename(
        columns={"date": "ds", "sales": "y"}
    )

    train_df = prophet_df[prophet_df["ds"] <= SPLIT_DATE]
    test_df = prophet_df[
        (prophet_df["ds"] > SPLIT_DATE) & (prophet_df["ds"] <= TEST_END)
    ]

    # Initialize and fit Prophet
    # daily_seasonality=False because daily retail patterns are captured by weekly seasonality
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
    )
    model.fit(train_df)

    # Forecast the 28 validation days
    future = model.make_future_dataframe(periods=28, freq="D")
    forecast = model.predict(future)

    # Extract test period predictions
    test_forecast = forecast[
        (forecast["ds"] > SPLIT_DATE) & (forecast["ds"] <= TEST_END)
    ][["ds", "yhat"]].copy()

    test_forecast = test_forecast.rename(
        columns={"ds": "date", "yhat": "predicted_sales"}
    )
    test_forecast["series_id"] = series_id

    # Combine with actuals
    merged = test_forecast.merge(
        series_df[["date", "sales", "store_id", "item_id"]],
        on="date",
        how="left",
    )

    # Units cannot be negative
    merged["predicted_sales"] = np.maximum(0, merged["predicted_sales"])
    merged["absolute_error"] = np.abs(merged["sales"] - merged["predicted_sales"])

    all_predictions.append(merged)
    print(f"[{i}/{len(series_list)}] Completed series: {series_id}")

# Combine predictions across all series
results_df = pd.concat(all_predictions, ignore_index=True)

# 3. Calculate evaluation metrics
y_true = results_df["sales"]
y_pred = results_df["predicted_sales"]

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
total_sales = y_true.sum()
wape = (results_df["absolute_error"].sum() / total_sales * 100) if total_sales > 0 else 0

print("\n" + "=" * 50)
print("PROPHET BASELINE EVALUATION RESULTS")
print("=" * 50)
print(f"MAE:  {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"WAPE: {wape:.2f}%")

# 4. Save Prophet predictions and report
predictions_path = REPORTS_DIR / "prophet_test_predictions.csv"
results_df.to_csv(predictions_path, index=False)
print(f"\nSaved test predictions: {predictions_path}")

report = {
    "model": "Prophet",
    "dataset": "retail_sales_daily_top20.csv",
    "series_count": int(len(series_list)),
    "train_end_date": str(SPLIT_DATE.date()),
    "test_start_date": "2016-03-28",
    "test_end_date": str(TEST_END.date()),
    "mae": float(mae),
    "rmse": float(rmse),
    "wape_percent": float(wape),
    "note": "Prophet baseline with yearly and weekly seasonality on top 20 series.",
}

report_path = REPORTS_DIR / "prophet_model_evaluation.json"
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print(f"Saved evaluation report: {report_path}")