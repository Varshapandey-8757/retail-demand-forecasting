"""Train LightGBM, then store the 30-day backtest (predicted vs actual) and the
next-30-day forecast in the demand_forecasts table.

Usage:
    python -m pipelines.run_lightgbm_forecast                 # reads PostgreSQL
    python -m pipelines.run_lightgbm_forecast --csv data/processed/retail_sales_daily_top20.csv
    python -m pipelines.run_lightgbm_forecast --calendar data/raw/calendar.csv --no-db
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.ml.data import get_engine, load_sales, make_run_id, to_forecast_rows, write_forecasts
from app.ml.features import HORIZON, load_future_calendar
from app.ml.lgbm_forecaster import predict_future, save_artifact, train_lightgbm

REPORT_PATH = Path(__file__).resolve().parents[1] / "reports" / "lightgbm_evaluation.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", help="Read sales from this CSV instead of PostgreSQL")
    ap.add_argument("--calendar", help="M5 calendar.csv for future events and SNAP days")
    ap.add_argument("--no-db", action="store_true", help="Train and save the model, skip writing to the database")
    args = ap.parse_args()

    sales = load_sales(args.csv)
    print(f"Loaded {len(sales):,} rows for {sales['series_id'].nunique()} series")

    artifact, backtest, metrics = train_lightgbm(sales)
    print(json.dumps(metrics, indent=2))

    model_path = save_artifact(artifact)
    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"Model saved to {model_path}")

    future = predict_future(
        artifact, sales, horizon=HORIZON,
        future_calendar=load_future_calendar(args.calendar),
    )

    if args.no_db:
        return

    run_id = make_run_id("lightgbm")
    rows = [
        to_forecast_rows(backtest, run_id=run_id, model_name="lightgbm",
                         model_version=artifact["version"], is_backtest=True),
        to_forecast_rows(future, run_id=run_id, model_name="lightgbm",
                         model_version=artifact["version"], is_backtest=False),
    ]
    import pandas as pd
    written = write_forecasts(get_engine(), pd.concat(rows, ignore_index=True))
    print(f"Stored {written:,} forecast rows (run {run_id})")


if __name__ == "__main__":
    main()