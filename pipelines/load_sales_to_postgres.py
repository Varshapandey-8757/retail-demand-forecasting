
from pathlib import Path

import pandas as pd
from sqlalchemy.dialects.postgresql import insert

from app.db.database import SessionLocal
from app.db.models import DailySales


# Project paths
BASE_DIR = Path(__file__).resolve().parents[1]
CSV_PATH = BASE_DIR / "data" / "processed" / "retail_sales_daily_top20.csv"

# Read data in batches of 5,000 rows
BATCH_SIZE = 1000

# Expected CSV columns
CSV_COLUMNS = [
    "series_id",
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id",
    "d",
    "sales",
    "date",
    "wm_yr_wk",
    "weekday",
    "wday",
    "month",
    "year",
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
    "snap_CA",
    "snap_TX",
    "snap_WI",
    "sell_price",
]

INTEGER_COLUMNS = [
    "sales",
    "wm_yr_wk",
    "wday",
    "month",
    "year",
    "snap_CA",
    "snap_TX",
    "snap_WI",
]


def transform_data(df: pd.DataFrame) -> list[dict]:
    """Clean and transform one CSV batch."""

    # Convert date strings into Python date objects
    df["date"] = pd.to_datetime(
        df["date"], errors="raise"
    ).dt.date

    # Validate and convert integer columns
    for column in INTEGER_COLUMNS:
        df[column] = pd.to_numeric(
            df[column], errors="raise"
        ).astype("int64")

    # Convert sell_price into numeric values.
    # Missing values remain missing.
    df["sell_price"] = pd.to_numeric(
        df["sell_price"], errors="raise"
    )

    # Convert pandas NaN/NA values into Python None.
    # SQLAlchemy will store these as SQL NULL.
    df = df.astype(object).where(pd.notna(df), None)

    return df[CSV_COLUMNS].to_dict(orient="records")


def load_sales_data():
    """Read the CSV and insert records into PostgreSQL."""

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"CSV file not found: {CSV_PATH}"
        )

    total_read = 0
    total_inserted = 0

    print("Starting M5 sales data ETL...")
    print(f"Source file: {CSV_PATH}")

    chunks = pd.read_csv(
        CSV_PATH,
        usecols=CSV_COLUMNS,
        chunksize=BATCH_SIZE,
    )

    for batch_number, df in enumerate(chunks, start=1):
        records = transform_data(df)

        total_read += len(records)

        # PostgreSQL INSERT with duplicate protection
        statement = (
            insert(DailySales)
            .values(records)
            .on_conflict_do_nothing(
                index_elements=["series_id", "d"]
            )
        )

        with SessionLocal() as db:
            try:
                result = db.execute(statement)
                db.commit()

                inserted = result.rowcount or 0
                total_inserted += inserted

            except Exception:
                db.rollback()
                raise

        print(
            f"Batch {batch_number}: "
            f"Read {len(records)}, inserted {inserted}"
        )

    print("\nETL completed successfully!")
    print(f"Total records read: {total_read}")
    print(f"New records inserted: {total_inserted}")
    print(f"Duplicate records skipped: {total_read - total_inserted}")


if __name__ == "__main__":
    load_sales_data()
