from datetime import date, datetime, timedelta
from pathlib import Path
import pandas as pd

from app.schemas.forecast import ForecastRecord

BASE_DIR = Path(__file__).resolve().parents[2]
PROPHET_FORECAST_FILE = BASE_DIR / "reports" / "prophet_future_30d_forecasts.csv"


def get_forecasts(
    store_id: str | None = None,
    item_id: str | None = None,
) -> list[ForecastRecord]:
    # Check if real Prophet 30-day forecast file exists
    if PROPHET_FORECAST_FILE.exists():
        df = pd.read_csv(PROPHET_FORECAST_FILE)

        if store_id:
            df = df[df["store_id"] == store_id]

        if item_id:
            df = df[df["item_id"] == item_id]

        records = [
            ForecastRecord(
                store_id=row["store_id"],
                item_id=row["item_id"],
                forecast_date=datetime.strptime(str(row["forecast_date"]), "%Y-%m-%d").date(),
                predicted_units=float(row["predicted_units"]),
                model_version=str(row["model_version"]),
            )
            for _, row in df.iterrows()
        ]
        return records

    # Fallback to demonstration data if pipeline has not been executed yet
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
