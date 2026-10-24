
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
file_path = BASE / "data" / "processed" / "retail_sales_daily_top20.csv"

df = pd.read_csv(file_path, parse_dates=["date"])

print("Total daily records:", len(df))
print("Unique product-store series:", df["series_id"].nunique())

print("\nTotal units sold:", df["sales"].sum())
print("Average daily units:", round(df["sales"].mean(), 2))

print("\nSales by store:")
print(
    df.groupby("store_id")["sales"]
    .sum()
    .sort_values(ascending=False)
)

print("\nSales by month:")
print(
    df.groupby(df["date"].dt.to_period("M"))["sales"]
    .sum()
    .tail(12)
)

print("\nDaily sales summary:")
print(df["sales"].describe())
