"""'What-if' price scenarios using the trained LightGBM model."""
from __future__ import annotations

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import DailySales
from app.ml.features import REQUIRED_COLUMNS
from app.ml.lgbm_forecaster import MODEL_PATH, load_artifact, predict_future
from app.schemas.forecast import WhatIfDay, WhatIfRequest, WhatIfResponse

HISTORY_DAYS = 240  # enough for the 56-day lags + 90-day rolling window + 30-day shift

_cache: dict = {"mtime": None, "artifact": None}


def load_lightgbm_artifact() -> dict:
    """Load the model once; reload automatically if the file is retrained."""
    mtime = MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else None
    if _cache["artifact"] is None or _cache["mtime"] != mtime:
        _cache["artifact"] = load_artifact()
        _cache["mtime"] = mtime
    return _cache["artifact"]


def fetch_history(db: Session, store_id: str, item_id: str, days: int = HISTORY_DAYS) -> pd.DataFrame:
    rows = db.execute(
        select(DailySales)
        .where(DailySales.store_id == store_id, DailySales.item_id == item_id)
        .order_by(DailySales.date.desc())
        .limit(days)
    ).scalars().all()
    data = [{c: getattr(r, c) for c in REQUIRED_COLUMNS} for r in rows]
    return pd.DataFrame(data, columns=REQUIRED_COLUMNS).sort_values("date").reset_index(drop=True)


def _pct(new: float, old: float) -> float | None:
    return None if old == 0 else round((new / old - 1) * 100, 2)


def run_what_if(db: Session, req: WhatIfRequest) -> WhatIfResponse:
    artifact = load_lightgbm_artifact()  # FileNotFoundError -> 503 in the route
    history = fetch_history(db, req.store_id, req.item_id)
    if history.empty:
        raise LookupError(f"No sales history for store {req.store_id}, item {req.item_id}.")
    if history["sell_price"].notna().sum() == 0:
        raise LookupError("This item has no price history, so a price scenario is not possible.")

    multiplier = 1 + req.price_change_pct / 100
    base = predict_future(artifact, history, 1.0, horizon=req.horizon_days)
    scen = predict_future(artifact, history, multiplier, horizon=req.horizon_days)

    merged = base.merge(scen, on="date", suffixes=("_base", "_scen"))
    daily = [
        WhatIfDay(
            date=r.date.date(),
            baseline_units=round(r.predicted_units_base, 2),
            scenario_units=round(r.predicted_units_scen, 2),
            baseline_price=round(r.sell_price_base, 2),
            scenario_price=round(r.sell_price_scen, 2),
        )
        for r in merged.itertuples()
    ]
    tb = float(sum(d.baseline_units for d in daily))
    ts = float(sum(d.scenario_units for d in daily))
    rb = float(sum(d.baseline_units * d.baseline_price for d in daily))
    rs = float(sum(d.scenario_units * d.scenario_price for d in daily))

    low, high = artifact["price_ratio_range"]
    ratios = scen["price_ratio"].dropna()
    outside = bool(len(ratios) and (ratios.min() < low or ratios.max() > high))
    warning = None
    if outside:
        warning = (
            "This price change is outside the price range the model saw in training "
            f"(price ratio {low:.2f}-{high:.2f}). Treat the result as a rough estimate."
        )

    return WhatIfResponse(
        store_id=req.store_id, item_id=req.item_id,
        price_change_pct=req.price_change_pct, horizon_days=req.horizon_days,
        baseline_price=daily[0].baseline_price, scenario_price=daily[0].scenario_price,
        total_baseline_units=round(tb, 2), total_scenario_units=round(ts, 2),
        units_change_pct=_pct(ts, tb),
        baseline_revenue=round(rb, 2), scenario_revenue=round(rs, 2),
        revenue_change_pct=_pct(rs, rb),
        outside_training_price_range=outside, warning=warning,
        model_version=artifact["version"], daily=daily,
    )