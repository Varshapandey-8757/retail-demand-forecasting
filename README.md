# Retail Demand Forecasting & Inventory Optimization

Retail Demand Forecasting & Inventory Optimization platform built with Python, FastAPI, PostgreSQL, dbt, Streamlit, and Machine Learning (Prophet & Random Forest).

---

## 🏗️ Project Architecture

```text
Raw Sales Data (Kaggle M5)
        │
        ▼
PostgreSQL / dbt Data Warehouse (retail_demand_dbt/)
        │
        ├──────────────────────────┐
        ▼                          ▼
Prophet Forecasting (Model A)   Random Forest Baseline
(pipelines/generate_prophet_forecasts.py)
        │                          │
        └─────────────┬────────────┘
                      ▼
             FastAPI REST Backend (app/)
                      │
                      ▼
         Streamlit Analytics Dashboard (dashboard/)
```

---

## 👥 Team & Roles

- **Varsha Pandey (Lead)**: FastAPI backend, PostgreSQL ETL, Random Forest baseline, Streamlit dashboard.
- **Ravi (Model A Lead)**: Prophet demand forecasting pipeline, US holiday modeling, 30-day forecast generation, unit test suites.
- **Abhishek**: dbt models (staging & marts).
- **Darshan**: Data warehouse ingestion.
- **Harshita**: LightGBM model.
- **Sairaj**: Dashboard and inventory optimization workflow.

---

## 🚀 Prophet Forecasting Module

- **Configuration**: `configs/prophet_config.json`
- **Pipelines**: `pipelines/generate_prophet_forecasts.py`
- **Notebooks**: `notebooks/05_prophet_baseline.py`, `notebooks/06_prophet_with_holidays.py`, `notebooks/07_compare_models.py`
- **Reports**: `reports/prophet_future_30d_forecasts.csv`, `reports/model_comparison_report.json`
- **Documentation**: [docs/prophet_forecasting_guide.md](docs/prophet_forecasting_guide.md)

### Running the Prophet Pipeline & Tests:
```powershell
# Generate 30-day forward predictions
python pipelines/generate_prophet_forecasts.py

# Run unit tests (9 passed)
python -m pytest tests/test_prophet_forecasts.py tests/test_forecast_service.py -v
```
