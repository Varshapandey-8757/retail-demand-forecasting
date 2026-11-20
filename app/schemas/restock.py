from datetime import date

from pydantic import BaseModel, Field


class IncomingShipment(BaseModel):
    days_from_now: int = Field(ge=0, le=60, description="0 = arrives on the first forecast day")
    quantity: float = Field(gt=0)


class RestockPlanRequest(BaseModel):
    store_id: str
    item_id: str
    current_stock: float = Field(ge=0)
    incoming: list[IncomingShipment] = Field(default_factory=list)
    lead_time_days: int = Field(default=2, ge=0, le=30)
    review_period_days: int = Field(default=7, ge=1, le=30)
    service_level: float = Field(default=0.95, gt=0.5, lt=0.999)


class PlannedOrder(BaseModel):
    order_date: date
    arrival_date: date
    quantity: int


class StockDay(BaseModel):
    date: date
    forecast_units: float
    end_of_day_stock: float


class RestockPlanResponse(BaseModel):
    store_id: str
    item_id: str
    model_version: str
    safety_stock: float
    safety_stock_basis: str
    orders: list[PlannedOrder]
    total_ordered_units: int
    projected_shortage_units: float
    schedule: list[StockDay]