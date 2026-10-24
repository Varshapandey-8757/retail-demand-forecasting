
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASE = Path(__file__).resolve().parents[1]
DATA_FILE = BASE / "data" / "processed" / "retail_sales_daily_top20.csv"
MODEL_DIR = BASE / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_FILE, parse_dates=["date"])
df = df.sort_values(["series_id", "date"]).copy()

# Previous sales se features banao
grouped = df.groupby("series_id")["sales"]

df["lag_1"] = grouped.shift(1)
df["lag_7"] = grouped.shift(7)
df["lag_28"] = grouped.shift(28)

# Past sales ka average; current day's sales include nahi hogi
df["rolling_mean_7"] = grouped.transform(
    lambda s: s.shift(1).rolling(7).mean()
)
df["rolling_mean_28"] = grouped.transform(
    lambda s: s.shift(1).rolling(28).mean()
)

# Calendar features
df["day_of_week"] = df["date"].dt.dayofweek
df["month"] = df["date"].dt.month
df["day_of_month"] = df["date"].dt.day

features = [
    "lag_1", "lag_7", "lag_28",
    "rolling_mean_7", "rolling_mean_28",
    "day_of_week", "month", "day_of_month"
]

df = df.dropna(subset=features + ["sales"]).copy()

# Last 28 days ko test data rakho
cutoff = df["date"].max() - pd.Timedelta(days=27)

train = df[df["date"] < cutoff]
test = df[df["date"] >= cutoff]

if train.empty or test.empty:
    raise ValueError("Training ya test data empty hai.")

X_train = train[features]
y_train = train["sales"]

X_test = test[features]
y_test = test["sales"]

print("Training records:", len(train))
print("Testing records:", len(test))
print("Test begins:", cutoff.date())

# Model train karo
model = RandomForestRegressor(
    n_estimators=100,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Test data par prediction
predictions = np.maximum(0, model.predict(X_test))

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

total_actual = y_test.sum()
wape = (
    np.abs(y_test.to_numpy() - predictions).sum() / total_actual * 100
    if total_actual > 0
    else None
)

print("\nEvaluation results")
print("MAE:", round(mae, 3))
print("RMSE:", round(rmse, 3))
print("WAPE (%):", round(wape, 2) if wape is not None else "N/A")

# Model save karo
model_path = MODEL_DIR / "retail_demand_baseline.joblib"
joblib.dump(model, model_path)

# Evaluation report save karo
report = {
    "model": "RandomForestRegressor",
    "series_count": int(df["series_id"].nunique()),
    "train_records": int(len(train)),
    "test_records": int(len(test)),
    "test_start_date": str(cutoff.date()),
    "test_end_date": str(test["date"].max().date()),
    "mae": float(mae),
    "rmse": float(rmse),
    "wape_percent": float(wape) if wape is not None else None,
    "note": "Development sample: top 20 product-store series"
}

with open(MODEL_DIR / "evaluation.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print("\nModel saved:", model_path)
print("Evaluation report saved.")
