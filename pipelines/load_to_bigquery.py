"""Copy daily_sales and demand_forecasts from PostgreSQL into Google BigQuery.

Needs: GOOGLE_CLOUD_PROJECT and BIGQUERY_DATASET in .env, and credentials
(`gcloud auth application-default login` or GOOGLE_APPLICATION_CREDENTIALS).

    python -m pipelines.load_to_bigquery
"""
from __future__ import annotations

import os

import pandas as pd
from dotenv import load_dotenv
from google.cloud import bigquery

from app.ml.data import get_engine

TABLES = ["daily_sales", "demand_forecasts"]


def main() -> None:
    load_dotenv()
    project = os.getenv("GOOGLE_CLOUD_PROJECT")
    dataset = os.getenv("BIGQUERY_DATASET", "retail_analytics")
    if not project:
        raise RuntimeError("GOOGLE_CLOUD_PROJECT is missing. Check your .env file.")

    client = bigquery.Client(project=project)
    client.create_dataset(bigquery.Dataset(f"{project}.{dataset}"), exists_ok=True)
    engine = get_engine()

    for table in TABLES:
        df = pd.read_sql_table(table, engine)
        job = client.load_table_from_dataframe(
            df, f"{project}.{dataset}.{table}",
            job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE"),
        )
        job.result()
        print(f"Loaded {len(df):,} rows into {dataset}.{table}")


if __name__ == "__main__":
    main()