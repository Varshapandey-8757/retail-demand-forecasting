"""LightGBM demand model: training, 30-day forecasting and what-if scenarios."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from app.ml.features import (
    CATEGORICAL, FEATURES, HORIZON, MONOTONE_CONSTRAINTS, apply_categories,
    build_features, learn_categories, prepare_frame,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "lightgbm_demand.joblib"

DEFAULT_PARAMS = dict(
    objective="poisson",  # sales are non-negative counts
    learning_rate=0.05,
    num_leaves=31,
    min_data_in_leaf=20,
    bagging_fraction=0.8,
    bagging_freq=1,
    feature_fraction=0.8,
    verbosity=-1,
    monotone_constraints=MONOTONE_CONSTRAINTS,  # price up -> demand never up
    monotone_constraints_method="intermediate",
)


def regression_metrics(actual, predicted) -> dict[str, float]:
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    err = a - p
    total = np.abs(a).sum()
    return {
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "wape_percent": float(np.abs(err).sum() / total * 100) if total else float("nan"),
        "rows": int(len(a)),
    }


def _model_frame(feats: pd.DataFrame, categories: dict) -> pd.DataFrame:
    return apply_categories(feats, categories)[FEATURES]


def train_lightgbm(
    raw: pd.DataFrame,
    valid_days: int = HORIZON,
    params: dict | None = None,
    seed: int = 42,
    max_rounds: int = 2000,
) -> tuple[dict, pd.DataFrame, dict]:
    """Train on history, score on the last `valid_days`, then refit on everything.

    Returns (artifact, backtest_frame, metrics). The backtest frame holds
    predicted vs actual for the held-out days (a true 30-day-ahead test).
    """
    frame = prepare_frame(raw)
    feats = build_features(frame)
    usable = feats[feats["sell_price"].notna() & feats["sales"].notna() & feats["lag_35"].notna()]

    last_date = usable["date"].max()
    cutoff = last_date - pd.Timedelta(days=valid_days)
    train = usable[usable["date"] <= cutoff]
    valid = usable[usable["date"] > cutoff]
    if train.empty or valid.empty:
        raise ValueError("Not enough history for a train/validation split.")

    categories = learn_categories(usable)
    params = {**DEFAULT_PARAMS, **(params or {}), "seed": seed}

    def dataset(part: pd.DataFrame) -> lgb.Dataset:
        return lgb.Dataset(
            _model_frame(part, categories), label=part["sales"],
            categorical_feature=CATEGORICAL, free_raw_data=False,
        )

    model = lgb.train(
        params, dataset(train), num_boost_round=max_rounds,
        valid_sets=[dataset(valid)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    best_rounds = max(int(model.best_iteration or max_rounds), 50)

    pred = np.clip(model.predict(_model_frame(valid, categories)), 0, None)
    baseline = valid["same_weekday_mean"].fillna(valid["roll_mean_28"]).fillna(0)
    metrics = {
        "model": "LightGBM",
        "validation": {
            "start": str(valid["date"].min().date()),
            "end": str(valid["date"].max().date()),
            "horizon_days": HORIZON,
            "note": "True 30-day-ahead evaluation: no feature uses sales from the last 30 days before the target day.",
        },
        "lightgbm": regression_metrics(valid["sales"], pred),
        "baseline_same_weekday_mean": regression_metrics(valid["sales"], baseline),
        "series_count": int(usable["series_id"].nunique()),
        "best_rounds": best_rounds,
    }

    # Refit on all usable rows with the number of rounds found above.
    final = lgb.train(params, dataset(usable), num_boost_round=best_rounds)

    ratio = usable["price_ratio"].dropna()
    artifact = {
        "model": final,
        "categories": categories,
        "features": FEATURES,
        "version": f"lightgbm-{dt.date.today():%Y%m%d}",
        "trained_through": str(last_date.date()),
        "horizon": HORIZON,
        "price_ratio_range": [float(ratio.quantile(0.01)), float(ratio.quantile(0.99))],
        "metrics": metrics,
    }

    backtest = valid[["series_id", "store_id", "item_id", "dept_id", "cat_id", "date", "sales"]].copy()
    backtest["predicted_units"] = pred
    backtest = backtest.rename(columns={"sales": "actual_units"})
    return artifact, backtest, metrics


def save_artifact(artifact: dict, path: Path = MODEL_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)
    return path


def load_artifact(path: Path = MODEL_PATH) -> dict:
    if not Path(path).exists():
        raise FileNotFoundError(f"LightGBM model not found: {path}")
    return joblib.load(path)


def predict_future(
    artifact: dict,
    history: pd.DataFrame,
    price_multiplier: float = 1.0,
    horizon: int = HORIZON,
    future_calendar: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Forecast the next `horizon` days after the end of `history`.

    `history` needs the raw sales columns (see features.REQUIRED_COLUMNS) and
    at least ~120 days per series. `price_multiplier` scales the future price
    (0.9 = a 10% price drop) - this is the what-if lever.
    """
    if not 1 <= horizon <= artifact["horizon"]:
        raise ValueError(f"horizon must be between 1 and {artifact['horizon']}")
    if price_multiplier <= 0:
        raise ValueError("price_multiplier must be positive")

    hist = prepare_frame(history)
    static_cols = ["series_id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    cal = None
    if future_calendar is not None:
        cal = future_calendar.set_index("date")

    pieces = [hist.drop(columns=["snap", "event_type"])]
    for sid, grp in hist.groupby("series_id", sort=False):
        last = grp.iloc[-1]
        priced = grp["sell_price"].dropna()
        if priced.empty:
            continue
        dates = pd.date_range(last["date"] + pd.Timedelta(days=1), periods=horizon, freq="D")
        fut = pd.DataFrame({"date": dates})
        for col in static_cols:
            fut[col] = last[col]
        fut["sales"] = np.nan
        fut["sell_price"] = float(priced.iloc[-1]) * price_multiplier
        for col in ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]:
            fut[col] = fut["date"].map(cal[col]) if cal is not None else None
        for col in ["snap_CA", "snap_TX", "snap_WI"]:
            fut[col] = fut["date"].map(cal[col]).fillna(0) if cal is not None else 0
        pieces.append(fut)

    combined = pd.concat(pieces, ignore_index=True)
    combined = prepare_frame(combined)
    feats = build_features(combined)
    origin = hist.groupby("series_id")["date"].max().rename("origin")
    feats = feats.join(origin, on="series_id")
    future = feats[feats["date"] > feats["origin"]].copy()

    future["predicted_units"] = np.clip(
        artifact["model"].predict(_model_frame(future, artifact["categories"])), 0, None
    )
    return future[["series_id", "store_id", "item_id", "dept_id", "cat_id",
                   "date", "predicted_units", "sell_price", "price_ratio"]].reset_index(drop=True)