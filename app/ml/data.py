"""Load sales history and write forecasts (PostgreSQL via DATABASE_URL, or CSV)."""
from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from app.db.models import DemandForecast

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = PROJECT_ROOT / "data" / "processed" / "retail_sales_daily_top20.csv"

SALES_COLUMNS = [
    "series_id", "item_id", "dept_id", "cat_id", "store_id", "state_id",
    "date", "sales", "sell_price",
    "event_name_1", "event_type_1", "event_name_2", "event_type_2",
    "snap_CA", "snap_TX", "snap_WI",
]


def get_engine() -> Engine:
    load_dotenv()
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is missing. Check your .env file.")
    return create_engine(url, pool_pre_ping=True)


def load_sales(csv_path: str | Path | None = None) -> pd.DataFrame:
    """Read daily sales from a CSV if given, otherwise from the daily_sales table."""
    if csv_path:
        df = pd.read_csv(csv_path)
    else:
        df = pd.read_sql_table("daily_sales", get_engine())
    return df[SALES_COLUMNS].copy()


def make_run_id(prefix: str) -> str:
    return f"{prefix}-{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%S}"


def write_forecasts(engine: Engine, frame: pd.DataFrame) -> int:
    """Append forecast rows to demand_forecasts (creating the table if needed).

    `frame` needs: run_id, model_name, model_version, series_id, store_id,
    item_id, dept_id, cat_id, forecast_date, predicted_units, actual_units,
    is_backtest.
    """
    DemandForecast.__table__.create(engine, checkfirst=True)
    cols = [c.name for c in DemandForecast.__table__.columns if c.name not in ("id", "created_at")]
    out = frame[cols].copy()
    out["forecast_date"] = pd.to_datetime(out["forecast_date"]).dt.date
    out.to_sql("demand_forecasts", engine, if_exists="append", index=False,
               method="multi", chunksize=2000)
    return len(out)


def to_forecast_rows(df: pd.DataFrame, *, run_id: str, model_name: str,
                     model_version: str, is_backtest: bool) -> pd.DataFrame:
    """Normalise model output (date/predicted_units[/actual_units]) to table rows."""
    out = df.rename(columns={"date": "forecast_date"}).copy()
    if "actual_units" not in out.columns:
        out["actual_units"] = float("nan")
    out["run_id"] = run_id
    out["model_name"] = model_name
    out["model_version"] = model_version
    out["is_backtest"] = is_backtest
    return out