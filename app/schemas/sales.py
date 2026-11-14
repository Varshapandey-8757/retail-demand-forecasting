
from datetime import date

from pydantic import BaseModel, ConfigDict


class SalesRecord(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    series_id: str
    item_id: str
    store_id: str
    date: date
    sales: int
    sell_price: float | None
    weekday: str
    month: int
    year: int


class SalesResponse(BaseModel):
    total_records: int
    limit: int
    offset: int
    sales: list[SalesRecord]
