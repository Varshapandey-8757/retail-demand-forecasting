
from pathlib import Path
import pandas as pd

# Project ke main folder se paths
BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "data" / "raw" / "m5-forecasting-accuracy"
PROCESSED = BASE / "data" / "processed"

PROCESSED.mkdir(parents=True, exist_ok=True)

sales_file = RAW / "sales_train_validation.csv"
calendar_file = RAW / "calendar.csv"
prices_file = RAW / "sell_prices.csv"

# 1. Files check karo
for file in [sales_file, calendar_file, prices_file]:
    if not file.exists():
        raise FileNotFoundError(f"File nahi mili: {file}")

print("All input files found.")

# 2. Sales data read karo
sales = pd.read_csv(sales_file)
print("Original sales shape:", sales.shape)

# Daily sales columns identify karo
day_cols = [c for c in sales.columns if c.startswith("d_")]

# 3. Har product-store combination ki total sales
sales["total_sales"] = sales[day_cols].sum(axis=1)

# Development ke liye top 20 combinations select karo
top_sales = sales.nlargest(20, "total_sales").copy()

print("\nSelected product-store combinations:", len(top_sales))

# 4. Wide format ko daily rows mein convert karo
id_cols = [
    "id", "item_id", "dept_id",
    "cat_id", "store_id", "state_id"
]

daily = top_sales.melt(
    id_vars=id_cols,
    value_vars=day_cols,
    var_name="d",
    value_name="sales"
)

daily = daily.rename(columns={"id": "series_id"})

# 5. Calendar connect karo
calendar = pd.read_csv(calendar_file)

calendar["date"] = pd.to_datetime(calendar["date"])

daily = daily.merge(
    calendar,
    on="d",
    how="left",
    validate="many_to_one"
)

# Calendar join verify karo
if daily["date"].isnull().any():
    raise ValueError(
        "Some sales dates were not matched with the calendar."
    )

print("Calendar dates matched successfully.")

# 6. Product prices connect karo
prices = pd.read_csv(prices_file)

daily = daily.merge(
    prices,
    on=["store_id", "item_id", "wm_yr_wk"],
    how="left",
    validate="many_to_one"
)

# 7. Date ke hisaab se sort karo
daily = daily.sort_values(
    ["series_id", "date"]
).reset_index(drop=True)

# 8. Basic checks
print("\nPrepared data shape:", daily.shape)
print("\nDate range:")
print(daily["date"].min(), "to", daily["date"].max())

print("\nMissing values:")
print(
    daily[["date", "sales", "sell_price"]]
    .isnull()
    .sum()
)

print("\nMissing values percentage:")
print(
    (daily[["date", "sales", "sell_price"]].isnull().mean() * 100)
    .round(2)
)

print("\nSample data:")
print(
    daily[
        ["date", "store_id", "item_id", "sales", "sell_price"]
    ].head(10).to_string(index=False)
)

# 9. Processed CSV save karo
output = PROCESSED / "retail_sales_daily_top20.csv"
daily.to_csv(output, index=False)

print("\nProcessed data saved at:")
print(output)
