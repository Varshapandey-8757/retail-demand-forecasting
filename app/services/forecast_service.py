"""Read model forecasts (and backtests) from the demand_forecasts table."""
from __future__ import annotations

from datetime import date, timedelta

from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
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
    try:
        row = db.execute(
            select(DemandForecast.run_id, DemandForecast.model_version)
            .where(DemandForecast.model_name == model_name)
            .order_by(DemandForecast.run_id.desc())
            .limit(1)
        ).first()
    except SQLAlchemyError:
        return None
    return (row[0], row[1]) if row else None


def demo_forecasts(
    *,
    model_name: str,
    group_by: str,
    store_id: str | None,
    item_id: str | None,
    dept_id: str | None,
    cat_id: str | None,
    start_date: date | None,
    end_date: date | None,
    include_backtest: bool,
    limit: int,
    offset: int,
) -> list[ForecastRecord]:
    """Return deterministic demo rows until a forecasting pipeline populates the database."""
    series = [
        ("STORE_001", "FOODS_1_001", "FOODS_1", "FOODS"),
        ("STORE_001", "FOODS_1_002", "FOODS_1", "FOODS"),
        ("STORE_002", "FOODS_1_001", "FOODS_1", "FOODS"),
        ("STORE_002", "HOUSEHOLD_1_001", "HOUSEHOLD_1", "HOUSEHOLD"),
    ]
    today = date.today()
    all_records: list[ForecastRecord] = []

    for s_store, s_item, s_dept, s_cat in series:
        if store_id and s_store != store_id:
            continue
        if item_id and s_item != item_id:
            continue
        if dept_id and s_dept != dept_id:
            continue
        if cat_id and s_cat != cat_id:
            continue

        day_count = 30
        days = []
        if include_backtest:
            days.extend(("backtest", today - timedelta(days=day_count - index), 17.0 + ((index + 2) % 7) * 2.2)
                        for index in range(day_count))
        days.extend(("future", today + timedelta(days=index), None) for index in range(day_count))

        for index, (row_type, forecast_day, actual) in enumerate(days):
            if start_date and forecast_day < start_date:
                continue
            if end_date and forecast_day > end_date:
                continue
            is_backtest = row_type == "backtest"
            predicted = 18.0 + (index % 7) * 2.5 + (3 if s_store.endswith("2") else 0)

            values: dict[str, str | None] = {}
            if group_by == "series":
                values = {"store_id": s_store, "item_id": s_item}
            elif group_by == "store":
                values = {"store_id": s_store}
            elif group_by == "dept":
                values = {"dept_id": s_dept}
            elif group_by == "category":
                values = {"cat_id": s_cat}

            all_records.append(ForecastRecord(
                **values,
                forecast_date=forecast_day,
                predicted_units=predicted,
                actual_units=actual,
                model_name=model_name,
                model_version="demo-v1",
                is_backtest=is_backtest,
            ))

    return all_records[offset:offset + limit]


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
        records = demo_forecasts(
            model_name=model_name,
            group_by=group_by,
            store_id=store_id,
            item_id=item_id,
            dept_id=dept_id,
            cat_id=cat_id,
            start_date=start_date,
            end_date=end_date,
            include_backtest=include_backtest,
            limit=limit,
            offset=offset,
        )
        return "demo", records
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