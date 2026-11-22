"""Prophet forecasts for the highest-volume series (trend, seasonality, events).

For each series: fit on history minus the last N days and score those days
(backtest), then refit on everything and forecast the next 30 days.
Results go to demand_forecasts with model_name='prophet'.

Usage:
    python -m pipelines.run_prophet_forecast --top-n 10
    python -m pipelines.run_prophet_forecast --csv data/processed/retail_sales_daily_top20.csv --calendar data/raw/calendar.csv
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd
from prophet import Prophet

from app.ml.data import get_engine, load_sales, make_run_id, to_forecast_rows, write_forecasts
from app.ml.features import HORIZON, load_future_calendar
from app.ml.lgbm_forecaster import regression_metrics

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)
MODEL_VERSION = "prophet-1.5"


def build_holidays(sales: pd.DataFrame, calendar: pd.DataFrame | None) -> pd.DataFrame | None:
    """Events from the data (and future calendar) as a Prophet holidays frame."""
    sources = [sales[["date", "event_name_1", "event_name_2"]]]
    if calendar is not None:
        sources.append(calendar[["date", "event_name_1", "event_name_2"]])
    ev = pd.concat(sources).drop_duplicates("date")
    parts = []
    for col in ("event_name_1", "event_name_2"):
        sub = ev[["date", col]].dropna().rename(columns={"date": "ds", col: "holiday"})
        parts.append(sub)
    hol = pd.concat(parts, ignore_index=True)
    if hol.empty:
        return None
    hol["ds"] = pd.to_datetime(hol["ds"])
    hol["lower_window"], hol["upper_window"] = 0, 0
    return hol.drop_duplicates(["ds", "holiday"])


def _fit(train: pd.DataFrame, holidays) -> Prophet:
    model = Prophet(
        yearly_seasonality=len(train) >= 730,
        weekly_seasonality=True,
        daily_seasonality=False,
        holidays=holidays,
    )
    model.add_regressor("sell_price", standardize=True)
    model.fit(train[["ds", "y", "sell_price"]])
    return model


def forecast_series(sdf: pd.DataFrame, holidays, horizon: int, valid_days: int):
    """Return (backtest_df, future_df) with columns date, predicted_units[, actual_units]."""
    d = sdf.sort_values("date").copy()
    d["sell_price"] = d["sell_price"].ffill().bfill()
    d = d.dropna(subset=["sell_price"]).rename(columns={"date": "ds", "sales": "y"})

    cutoff = d["ds"].max() - pd.Timedelta(days=valid_days)
    train, valid = d[d["ds"] <= cutoff], d[d["ds"] > cutoff]
    m = _fit(train, holidays)
    pred = m.predict(valid[["ds", "sell_price"]])
    backtest = pd.DataFrame({
        "date": valid["ds"].values,
        "predicted_units": np.clip(pred["yhat"].values, 0, None),
        "actual_units": valid["y"].values,
    })

    m_full = _fit(d, holidays)
    future_dates = pd.date_range(d["ds"].max() + pd.Timedelta(days=1), periods=horizon)
    fut = pd.DataFrame({"ds": future_dates, "sell_price": float(d["sell_price"].iloc[-1])})
    fpred = m_full.predict(fut)
    future = pd.DataFrame({"date": future_dates,
                           "predicted_units": np.clip(fpred["yhat"].values, 0, None)})
    return backtest, future


def run_prophet(sales: pd.DataFrame, top_n: int, horizon: int, valid_days: int,
                calendar: pd.DataFrame | None):
    sales = sales.copy()
    sales["date"] = pd.to_datetime(sales["date"])
    top = sales.groupby("series_id")["sales"].sum().nlargest(top_n).index
    holidays = build_holidays(sales, calendar)
    back_parts, fut_parts = [], []
    for sid in top:
        sdf = sales[sales["series_id"] == sid]
        meta = sdf.iloc[0][["series_id", "store_id", "item_id", "dept_id", "cat_id"]].to_dict()
        backtest, future = forecast_series(sdf, holidays, horizon, valid_days)
        for part, bucket in ((backtest, back_parts), (future, fut_parts)):
            for k, v in meta.items():
                part[k] = v
            bucket.append(part)
    return pd.concat(back_parts, ignore_index=True), pd.concat(fut_parts, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv")
    ap.add_argument("--calendar")
    ap.add_argument("--top-n", type=int, default=10, help="Number of highest-volume series")
    ap.add_argument("--horizon", type=int, default=HORIZON)
    ap.add_argument("--valid-days", type=int, default=HORIZON)
    ap.add_argument("--no-db", action="store_true")
    args = ap.parse_args()

    sales = load_sales(args.csv)
    backtest, future = run_prophet(sales, args.top_n, args.horizon, args.valid_days,
                                   load_future_calendar(args.calendar))
    print("Prophet backtest:", regression_metrics(backtest["actual_units"], backtest["predicted_units"]))
    if args.no_db:
        return

    run_id = make_run_id("prophet")
    rows = pd.concat([
        to_forecast_rows(backtest, run_id=run_id, model_name="prophet",
                         model_version=MODEL_VERSION, is_backtest=True),
        to_forecast_rows(future, run_id=run_id, model_name="prophet",
                         model_version=MODEL_VERSION, is_backtest=False),
    ], ignore_index=True)
    print(f"Stored {write_forecasts(get_engine(), rows):,} forecast rows (run {run_id})")


if __name__ == "__main__":
    main()