
from pathlib import Path

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert

from app.db.database import engine
from app.db.models import DailySales


# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Input CSV file
CSV_PATH = BASE_DIR / "data" / "processed" / "retail_sales_daily_top20.csv"

# Number of records processed in each batch
BATCH_SIZE = 1000


def load_daily_sales():
    print("Starting M5 sales data ETL...")

    # 1. EXTRACT
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    print(f"CSV records found: {len(df)}")

    # Check required columns
    expected_columns = [
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

    missing_columns = set(expected_columns) - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing CSV columns: {missing_columns}")

    # Keep only the required columns in the correct order
    df = df[expected_columns].copy()

    # 2. TRANSFORM

    # Convert date column into Python date objects
    df["date"] = pd.to_datetime(
        df["date"],
        errors="raise"
    ).dt.date

    # Convert missing values (NaN/NaT) to Python None
    # PostgreSQL will store these values as NULL
    df = df.astype(object).where(pd.notna(df), None)

    # Check duplicate keys in the source CSV
    duplicate_count = df.duplicated(
        subset=["series_id", "d"]
    ).sum()

    print(f"Duplicate source records: {duplicate_count}")

    if duplicate_count > 0:
        raise ValueError(
            "Duplicate records found in CSV for series_id and d."
        )

    # Convert DataFrame into a list of dictionaries
    records = df.to_dict(orient="records")

    # 3. LOAD
    with engine.begin() as connection:

        # Count existing database records before loading
        before_count = connection.execute(
            select(func.count()).select_from(DailySales)
        ).scalar_one()

        print(f"Existing database records: {before_count}")

        # Process records in batches
        for start in range(0, len(records), BATCH_SIZE):

            batch = records[start:start + BATCH_SIZE]

            # Prepare PostgreSQL INSERT statement
            statement = insert(DailySales).values(batch)

            # Skip records that already exist
            statement = statement.on_conflict_do_nothing(
                constraint="uq_daily_sales_series_day"
            )

            # Execute batch
            connection.execute(statement)

            processed = min(start + BATCH_SIZE, len(records))

            print(
                f"Processed {processed} / {len(records)} records"
            )

        # Count records after loading
        total_count = connection.execute(
            select(func.count()).select_from(DailySales)
        ).scalar_one()

        # Calculate actual newly inserted records
        inserted_count = total_count - before_count

    # 4. ETL SUMMARY
    print("\nETL completed successfully!")
    print(f"Source CSV records: {len(records)}")
    print(f"New records inserted: {inserted_count}")
    print(f"Total records in daily_sales: {total_count}")


if __name__ == "__main__":
    load_daily_sales()
