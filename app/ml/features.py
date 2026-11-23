"""Feature engineering shared by training and inference.

Design rule: every feature that depends on past *sales* is shifted by at
least HORIZON (30) days. That makes every feature known at the forecast
origin for any day in the next 30 days, so:

* a 30-day-ahead forecast needs no recursion (no error build-up),
* the validation score is a genuine 30-day-ahead score,
* "what-if" price scenarios only change the price columns.

Prices and calendar events are treated as known in advance (as in M5).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HORIZON = 30
LAGS = (35, 42, 49, 56)  # same weekday as the target day, all >= HORIZON
EVENT_TYPES = ["None", "Cultural", "National", "Religious", "Sporting"]
CATEGORICAL = ["store_id", "dept_id", "cat_id", "item_id", "event_type"]
NUMERIC = [
    "lag_35", "lag_42", "lag_49", "lag_56", "same_weekday_mean",
    "roll_mean_7", "roll_mean_28", "roll_mean_90", "roll_std_28",
    "price_ratio",  # relative price only: absolute price would act as an item/time ID
    "has_event", "snap",
    "day_of_week", "month", "day_of_month", "week_of_year", "is_weekend",
]
FEATURES = NUMERIC + CATEGORICAL

# Business rule enforced in the model: a higher relative price can never
# increase demand (-1 = non-increasing). All other features are unconstrained.
MONOTONE_CONSTRAINTS = [-1 if f == "price_ratio" else 0 for f in FEATURES]

REQUIRED_COLUMNS = [
    "series_id", "item_id", "dept_id", "cat_id", "store_id", "state_id",
    "date", "sales", "sell_price",
    "event_name_1", "event_type_1", "event_name_2", "event_type_2",
    "snap_CA", "snap_TX", "snap_WI",
]


def prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Validate raw sales rows and add the derived `snap`/`event_type` columns."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values(["series_id", "date"]).reset_index(drop=True)

    # Each series must be a gap-free daily series, because lags are row shifts.
    gaps = out.groupby("series_id")["date"].diff().dropna()
    if (gaps != pd.Timedelta(days=1)).any():
        raise ValueError(
            "Each series must have one row per day with no gaps or duplicates."
        )

    state = out["state_id"].astype(str)
    out["snap"] = np.select(
        [state == "CA", state == "TX", state == "WI"],
        [out["snap_CA"], out["snap_TX"], out["snap_WI"]],
        default=0,
    ).astype(float)

    types = out["event_type_1"].where(out["event_type_1"].notna(), out["event_type_2"])
    out["event_type"] = types.fillna("None").astype(str)
    out.loc[~out["event_type"].isin(EVENT_TYPES), "event_type"] = "None"
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add model features to a prepared, gap-free frame (future rows have NaN sales)."""
    df = df.sort_values(["series_id", "date"]).reset_index(drop=True).copy()
    g = df.groupby("series_id", sort=False)

    for lag in LAGS:
        df[f"lag_{lag}"] = g["sales"].shift(lag)
    df["same_weekday_mean"] = df[[f"lag_{lag}" for lag in LAGS]].mean(axis=1)

    def rolled(series: pd.Series, window: int, fn: str, min_periods: int) -> pd.Series:
        return getattr(series.shift(HORIZON).rolling(window, min_periods=min_periods), fn)()

    df["roll_mean_7"] = g["sales"].transform(lambda s: rolled(s, 7, "mean", 3))
    df["roll_mean_28"] = g["sales"].transform(lambda s: rolled(s, 28, "mean", 7))
    df["roll_mean_90"] = g["sales"].transform(lambda s: rolled(s, 90, "mean", 14))
    df["roll_std_28"] = g["sales"].transform(lambda s: rolled(s, 28, "std", 7))

    price_ref = g["sell_price"].transform(lambda s: rolled(s, 28, "mean", 1))
    df["price_ratio"] = df["sell_price"] / price_ref

    df["has_event"] = (df["event_name_1"].notna() | df["event_name_2"].notna()).astype(int)
    d = df["date"]
    df["day_of_week"] = d.dt.dayofweek
    df["month"] = d.dt.month
    df["day_of_month"] = d.dt.day
    df["week_of_year"] = d.dt.isocalendar().week.astype(int)
    df["is_weekend"] = (d.dt.dayofweek >= 5).astype(int)
    return df


def apply_categories(df: pd.DataFrame, categories: dict[str, list[str]]) -> pd.DataFrame:
    """Cast categorical columns with a fixed category list (unknown values -> NaN)."""
    out = df.copy()
    for col in CATEGORICAL:
        out[col] = pd.Categorical(out[col].astype(str), categories=categories[col])
    return out


def learn_categories(df: pd.DataFrame) -> dict[str, list[str]]:
    cats = {c: sorted(df[c].astype(str).unique().tolist()) for c in CATEGORICAL if c != "event_type"}
    cats["event_type"] = list(EVENT_TYPES)
    return cats


def load_future_calendar(path: str | None) -> pd.DataFrame | None:
    """Read the M5 calendar.csv (events + SNAP days) so future days get real events."""
    if not path:
        return None
    cal = pd.read_csv(path)
    cal["date"] = pd.to_datetime(cal["date"])
    keep = ["date", "event_name_1", "event_type_1", "event_name_2", "event_type_2",
            "snap_CA", "snap_TX", "snap_WI"]
    return cal[keep]