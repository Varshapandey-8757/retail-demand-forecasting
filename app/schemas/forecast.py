from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


class ForecastRecord(BaseModel):
    # Fields that are not part of the chosen grouping are None.
    store_id: str | None = None
    item_id: str | None = None
    dept_id: str | None = None
    cat_id: str | None = None
    forecast_date: date
    predicted_units: float = Field(ge=0)
    actual_units: float | None = None  # only for backtest days
    model_name: str
    model_version: str
    is_backtest: bool = False


class ForecastResponse(BaseModel):
    source: str
    model_name: str
    run_id: str | None
    group_by: str
    forecasts: list[ForecastRecord]


class WhatIfRequest(BaseModel):
    store_id: str
    item_id: str
    price_change_pct: float = Field(ge=-50, le=50, description="-10 means a 10% price drop")
    horizon_days: int = Field(default=30, ge=1, le=30)


class WhatIfDay(BaseModel):
    date: date
    baseline_units: float
    scenario_units: float
    baseline_price: float
    scenario_price: float


class WhatIfResponse(BaseModel):
    store_id: str
    item_id: str
    price_change_pct: float
    horizon_days: int
    baseline_price: float
    scenario_price: float
    total_baseline_units: float
    total_scenario_units: float
    units_change_pct: float | None
    baseline_revenue: float
    scenario_revenue: float
    revenue_change_pct: float | None
    outside_training_price_range: bool
    warning: str | None = None
    model_version: str
    daily: list[WhatIfDay]


GroupBy = Literal["series", "store", "dept", "category"]