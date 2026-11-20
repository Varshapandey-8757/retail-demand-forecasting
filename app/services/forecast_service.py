"""Read model forecasts (and backtests) from the demand_forecasts table."""
from __future__ import annotations

from datetime import date

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import DemandForecast
from app.schemas.forecast import ForecastRecord

GROUP_COLUMNS = {
    "series": ["store_id", "item_id"],
    "store": ["store_id"],
    "dept": ["dept_id"],
    "category": ["cat_id"],
}


def latest_run(db: Session, model_name: str) -> tuple[str, str] | None:
    """(run_id, model_version) of the newest run of a model. Run ids sort by time."""
    row = db.execute(
        select(DemandForecast.run_id, DemandForecast.model_version)
        .where(DemandForecast.model_name == model_name)
        .order_by(DemandForecast.run_id.desc())
        .limit(1)
    ).first()
    return (row[0], row[1]) if row else None


def get_forecasts(
    db: Session,
    *,
    model_name: str = "lightgbm",
    group_by: str = "series",
    store_id: str | None = None,
    item_id: str | None = None,
    dept_id: str | None = None,
    cat_id: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    include_backtest: bool = False,
    limit: int = 10000,
    offset: int = 0,
) -> tuple[str | None, list[ForecastRecord]]:
    run = latest_run(db, model_name)
    if run is None:
        return None, []
    run_id, version = run

    cols = [getattr(DemandForecast, c) for c in GROUP_COLUMNS[group_by]]
    query = (
        select(
            *cols,
            DemandForecast.forecast_date,
            DemandForecast.is_backtest,
            func.sum(DemandForecast.predicted_units).label("predicted"),
            func.sum(DemandForecast.actual_units).label("actual"),
        )
        .where(DemandForecast.run_id == run_id, DemandForecast.model_name == model_name)
        .group_by(*cols, DemandForecast.forecast_date, DemandForecast.is_backtest)
        .order_by(DemandForecast.forecast_date, *cols)
        .limit(limit)
        .offset(offset)
    )
    if not include_backtest:
        query = query.where(DemandForecast.is_backtest.is_(False))
    for column, value in (
        (DemandForecast.store_id, store_id), (DemandForecast.item_id, item_id),
        (DemandForecast.dept_id, dept_id), (DemandForecast.cat_id, cat_id),
    ):
        if value:
            query = query.where(column == value)
    if start_date:
        query = query.where(DemandForecast.forecast_date >= start_date)
    if end_date:
        query = query.where(DemandForecast.forecast_date <= end_date)

    records = []
    for row in db.execute(query).all():
        values = dict(zip(GROUP_COLUMNS[group_by], row[: len(cols)]))
        date_, is_bt, predicted, actual = row[len(cols):]
        records.append(ForecastRecord(
            **values,
            forecast_date=date_,
            predicted_units=max(0.0, float(predicted)),
            actual_units=None if actual is None else float(actual),
            model_name=model_name,
            model_version=version,
            is_backtest=bool(is_bt),
        ))
    return run_id, records


def backtest_residual_std(db: Session, store_id: str, item_id: str,
                          model_name: str = "lightgbm") -> float | None:
    """Std-dev of (actual - predicted) on backtest days for one series."""
    run = latest_run(db, model_name)
    if run is None:
        return None
    rows = db.execute(
        select(DemandForecast.predicted_units, DemandForecast.actual_units)
        .where(
            DemandForecast.run_id == run[0],
            DemandForecast.model_name == model_name,
            DemandForecast.store_id == store_id,
            DemandForecast.item_id == item_id,
            DemandForecast.is_backtest.is_(True),
            DemandForecast.actual_units.is_not(None),
        )
    ).all()
    if len(rows) < 5:
        return None
    errors = [a - p for p, a in rows]
    mean = sum(errors) / len(errors)
    return (sum((e - mean) ** 2 for e in errors) / (len(errors) - 1)) ** 0.5