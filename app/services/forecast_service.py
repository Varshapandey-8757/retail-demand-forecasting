
from datetime import date, timedelta

from app.schemas.forecast import ForecastRecord


def get_forecasts(
    store_id: str | None = None,
    item_id: str | None = None,
) -> list[ForecastRecord]:
    # Temporary demonstration data.
    # Replace this with model-generated forecasts or warehouse data later.
    today = date.today()

    records = [
        ForecastRecord(
            store_id="STORE_001",
            item_id="ITEM_001",
            forecast_date=today + timedelta(days=1),
            predicted_units=25,
            model_version="demo-v1",
        ),
        ForecastRecord(
            store_id="STORE_001",
            item_id="ITEM_002",
            forecast_date=today + timedelta(days=1),
            predicted_units=12,
            model_version="demo-v1",
        ),
        ForecastRecord(
            store_id="STORE_002",
            item_id="ITEM_001",
            forecast_date=today + timedelta(days=1),
            predicted_units=18,
            model_version="demo-v1",
        ),
    ]

    if store_id:
        records = [
            record for record in records
            if record.store_id == store_id
        ]

    if item_id:
        records = [
            record for record in records
            if record.item_id == item_id
        ]

    return records
