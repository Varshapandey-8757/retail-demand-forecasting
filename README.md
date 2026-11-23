# Retail Demand Forecasting & Inventory Optimization

An end-to-end retail analytics platform for sales analysis, demand forecasting,
restocking recommendations, and price what-if scenarios.

## Features

- FastAPI endpoints for health, sales, forecasts, inventory, restocking, and what-if analysis
- Streamlit executive dashboard with date/store/product filters and downloadable reports
- LightGBM and Prophet forecasting pipelines
- PostgreSQL/SQLite database support
- dbt staging and mart models for warehouse analytics
- Optional viewer/admin API-key authentication

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Set `DATABASE_URL` in `.env`. For a local database-free demo:

```text
DATABASE_URL=sqlite://
```

Without `VIEWER_API_KEY` and `ADMIN_API_KEY`, API authentication is disabled.
To enable it, set both keys in `.env` and use `X-API-Key` in API requests.

## Run

Start the API:

```powershell
python -m uvicorn app.main:app --reload
```

Start the dashboard:

```powershell
python -m streamlit run dashboard/app.py
```

- API: http://127.0.0.1:8000
- Swagger: http://127.0.0.1:8000/docs
- Dashboard: http://localhost:8501

## Forecast pipelines

```powershell
python -m pipelines.run_lightgbm_forecast
python -m pipelines.run_prophet_forecast
```

The API serves demo forecasts when no demand_forecasts table or model run exists.

## Tests

```powershell
python -m pytest -q
```

## dbt

The warehouse project is under `retail_demand_dbt/`.

```powershell
copy profiles.yml.example $env:USERPROFILE\.dbt\profiles.yml
dbt build --project-dir retail_demand_dbt --profiles-dir $env:USERPROFILE\.dbt
```

Set `DBT_POSTGRES_PASSWORD` and edit `profiles.yml` for your database credentials.
Set `DBT_SOURCE_SCHEMA` when the raw tables are outside `public`.

Key marts include daily sales, weekly/monthly sales, department planning, and
forecast accuracy.
