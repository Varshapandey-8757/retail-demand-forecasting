# Prophet Demand Forecasting Guide (Model A)

This document explains the Prophet forecasting pipeline, model evaluation, and backend integration for the Retail Demand Forecasting & Inventory Optimization project.

---

## 1. Overview & Architecture

Prophet acts as the forward-looking time series forecasting engine. Unlike standard regression models that require known daily lag features, Prophet fits trend, weekly seasonality, yearly seasonality, and holiday events to project daily demand 30 days into the future.

```text
Processed Data (data/processed/retail_sales_daily_top20.csv)
                      │
                      ▼
Prophet Pipeline (pipelines/generate_prophet_forecasts.py)
                      │
                      ▼
30-Day Export (reports/prophet_future_30d_forecasts.csv)
                      │
                      ▼
FastAPI Service (app/services/forecast_service.py)
                      │
                      ▼
Streamlit Dashboard (/api/v1/forecasts -> dashboard/app.py)
```

---

## 2. Directory & File Structure

- `configs/prophet_config.json`: Central configuration containing validation cutoff dates, series count, horizon length, and output schemas.
- `notebooks/05_prophet_baseline.py`: Baseline training and 28-day chronological validation.
- `notebooks/06_prophet_with_holidays.py`: Model experiment adding US country holidays.
- `notebooks/07_compare_models.py`: Side-by-side metric comparison between Random Forest and Prophet.
- `pipelines/generate_prophet_forecasts.py`: Production script generating the next 30 days of sales predictions for all 20 series.
- `reports/prophet_future_30d_forecasts.csv`: Exported 30-day forecast dataset (600 records).
- `reports/model_comparison_report.json`: Formal evaluation metrics and comparison breakdown.
- `tests/test_prophet_forecasts.py`: Unit tests validating output schema, zero nulls, non-negative bounds, and record counts.
- `tests/test_forecast_service.py`: Unit tests validating FastAPI service integration and store/item filtering.

---

## 3. Validation Methodology & Metrics

To prevent data leakage, training and validation follow a strict chronological split:
- **Training Period**: History up to `2016-03-27`
- **Validation Period**: 28 unobserved days from `2016-03-28` to `2016-04-24`

### Benchmark Results (Top 20 Series)

| Model | MAE | RMSE | WAPE (%) | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Random Forest Baseline** | 10.569 | 14.766 | 19.74% | Requires historical lag values for each test day |
| **Prophet Baseline** | 13.115 | 18.529 | 24.50% | Pure forward-looking time series |
| **Prophet + US Holidays** | **13.078** | **18.473** | **24.42%** | Captures national holiday spikes and closures |

---

## 4. How to Run the Pipeline

### Step 1: Generate 30-day future forecasts
```powershell
python pipelines/generate_prophet_forecasts.py
```

### Step 2: Run test suites
```powershell
python -m pytest tests/test_prophet_forecasts.py -v
python -m pytest tests/test_forecast_service.py -v
```

### Step 3: Run model comparison
```powershell
python notebooks/07_compare_models.py
```

---

## 5. API & Integration Contract

The forecast output strictly matches `app.schemas.forecast.ForecastRecord`:

| Field | Type | Description |
| :--- | :--- | :--- |
| `store_id` | `str` | Store identifier (e.g. `CA_1`) |
| `item_id` | `str` | Product identifier (e.g. `FOODS_3_090`) |
| `forecast_date` | `date` | Forecast date (`YYYY-MM-DD`) |
| `predicted_units` | `float` | Projected demand units (strictly $\ge 0$) |
| `model_version` | `str` | Tagged version identifier (`prophet-v1`) |
