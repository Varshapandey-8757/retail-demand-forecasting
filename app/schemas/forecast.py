
from datetime import date

from pydantic import BaseModel, Field


class ForecastRecord(BaseModel):
    store_id: str
    item_id: str
    forecast_date: date
    predicted_units: float = Field(ge=0)
    model_version: str


class ForecastResponse(BaseModel):
    source: str
    forecasts: list[ForecastRecord]
