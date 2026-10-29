
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE = Path(__file__).resolve().parents[1]

DATA_FILE = (
    BASE
    / "data"
    / "processed"
    / "retail_sales_daily_top20.csv"
)

MODEL_DIR = BASE / "models"
REPORT_DIR = BASE / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. LOAD AND VALIDATE DATA
# ==========================================

print("=" * 60)
print("RETAIL DEMAND FORECASTING - MODEL TRAINING")
print("=" * 60)

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_FILE}\n"
        "Run notebooks/02_prepare_sales_data.py first."
    )

df = pd.read_csv(DATA_FILE, parse_dates=["date"])

required_columns = ["date", "series_id", "sales"]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

df["sales"] = pd.to_numeric(df["sales"], errors="raise")

if df[["date", "series_id", "sales"]].isnull().any().any():
    raise ValueError(
        "Date, series_id, or sales contains missing values."
    )

if (df["sales"] < 0).any():
    raise ValueError("Sales contains negative values.")

df = df.sort_values(["series_id", "date"]).copy()

print("\nDataset loaded successfully.")
print("Total records:", len(df))
print("Product-store series:", df["series_id"].nunique())
print("Date range:", df["date"].min().date(),
      "to", df["date"].max().date())


# ==========================================
# 3. CREATE LAG AND ROLLING FEATURES
# ==========================================

grouped = df.groupby("series_id")["sales"]

# Previous sales values
df["lag_1"] = grouped.shift(1)
df["lag_7"] = grouped.shift(7)
df["lag_28"] = grouped.shift(28)

# Historical averages; current day's sales excluded
df["rolling_mean_7"] = grouped.transform(
    lambda s: s.shift(1).rolling(window=7).mean()
)

df["rolling_mean_28"] = grouped.transform(
    lambda s: s.shift(1).rolling(window=28).mean()
)

# Calendar features
df["day_of_week"] = df["date"].dt.dayofweek
df["month"] = df["date"].dt.month
df["day_of_month"] = df["date"].dt.day

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

df = df.dropna(subset=FEATURES + ["sales"]).copy()

if df.empty:
    raise ValueError(
        "No usable records remain after creating lag features."
    )


# ==========================================
# 4. TIME-BASED TRAIN / TEST SPLIT
# ==========================================

# Last 28 calendar days of each series are test data
last_date_by_series = df.groupby("series_id")["date"].transform("max")

df["is_test"] = (
    df["date"] >= last_date_by_series - pd.Timedelta(days=27)
)

train = df.loc[~df["is_test"]].copy()
test = df.loc[df["is_test"]].copy()

if train.empty or test.empty:
    raise ValueError("Training or testing dataset is empty.")

if train["date"].max() >= test["date"].min():
    print(
        "\nNote: Training and testing date ranges overlap "
        "because each series is split independently."
    )

X_train = train[FEATURES]
y_train = train["sales"]

X_test = test[FEATURES]
y_test = test["sales"]

print("\n" + "=" * 60)
print("TRAINING AND TESTING SUMMARY")
print("=" * 60)

print("Training records:", len(train))
print("Testing records:", len(test))
print("Training end date:", train["date"].max().date())
print("Testing date range:",
      test["date"].min().date(), "to", test["date"].max().date())


# ==========================================
# 5. TRAIN RANDOM FOREST MODEL
# ==========================================

print("\nTraining Random Forest model...")

model = RandomForestRegressor(
    n_estimators=100,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)

print("Model training completed.")


# ==========================================
# 6. EVALUATE MODEL
# ==========================================

predictions = np.maximum(0, model.predict(X_test))

mae = mean_absolute_error(y_test, predictions)

rmse = np.sqrt(
    mean_squared_error(y_test, predictions)
)

absolute_error_sum = np.abs(
    y_test.to_numpy() - predictions
).sum()

total_actual = y_test.sum()

wape = (
    absolute_error_sum / total_actual * 100
    if total_actual > 0
    else None
)

print("\n" + "=" * 60)
print("MODEL EVALUATION RESULTS")
print("=" * 60)

print("MAE:", round(mae, 3))
print("RMSE:", round(rmse, 3))
print(
    "WAPE (%):",
    round(wape, 2) if wape is not None else "N/A"
)


# ==========================================
# 7. SAVE MODEL
# ==========================================

model_path = MODEL_DIR / "retail_demand_baseline.joblib"

joblib.dump(model, model_path)

print("\nModel saved:", model_path)


# ==========================================
# 8. SAVE EVALUATION REPORT
# ==========================================

report = {
    "model": "RandomForestRegressor",
    "dataset": DATA_FILE.name,
    "series_count": int(df["series_id"].nunique()),
    "train_records": int(len(train)),
    "test_records": int(len(test)),
    "train_end_date": str(train["date"].max().date()),
    "test_start_date": str(test["date"].min().date()),
    "test_end_date": str(test["date"].max().date()),
    "mae": float(mae),
    "rmse": float(rmse),
    "wape_percent": (
        float(wape) if wape is not None else None
    ),
    "note": (
        "Baseline model using a development sample of "
        "top 20 product-store series. Evaluation uses "
        "historical lag features; it is not a recursive "
        "28-day-ahead forecast simulation."
    ),
}

model_report_path = MODEL_DIR / "evaluation.json"
public_report_path = REPORT_DIR / "model_evaluation.json"

for report_path in [model_report_path, public_report_path]:
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

print("Evaluation report saved:", model_report_path)
print("Project report saved:", public_report_path)


# ==========================================
# 9. SAVE TEST PREDICTIONS
# ==========================================

prediction_report = test[
    ["date", "series_id", "sales"]
].copy()

prediction_report["predicted_sales"] = predictions

prediction_report["absolute_error"] = np.abs(
    prediction_report["sales"]
    - prediction_report["predicted_sales"]
)

prediction_path = REPORT_DIR / "test_predictions.csv"

prediction_report.to_csv(prediction_path, index=False)

print("Test predictions saved:", prediction_path)


# ==========================================
# 10. FEATURE IMPORTANCE
# ==========================================

importance = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=False)

importance_path = REPORT_DIR / "feature_importance.csv"

importance.to_csv(importance_path, index=False)

print("\nFeature importance:")
print(importance.to_string(index=False))

print("\n" + "=" * 60)
print("MODEL TRAINING AND EVALUATION COMPLETED")
print("=" * 60)
