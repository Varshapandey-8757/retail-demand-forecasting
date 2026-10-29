
from pathlib import Path
import pandas as pd

# ==========================================
# 1. PROJECT PATH
# ==========================================

BASE = Path(__file__).resolve().parents[1]

file_path = (
    BASE
    / "data"
    / "processed"
    / "retail_sales_daily_top20.csv"
)

# ==========================================
# 2. LOAD DATA
# ==========================================

print("=" * 55)
print("RETAIL DEMAND FORECASTING - SALES ANALYSIS")
print("=" * 55)

if not file_path.exists():
    raise FileNotFoundError(
        f"Processed dataset not found: {file_path}\n"
        "Please run notebooks/02_prepare_sales_data.py first."
    )

df = pd.read_csv(file_path, parse_dates=["date"])

# Check required columns
required_columns = [
    "date",
    "series_id",
    "store_id",
    "item_id",
    "sales",
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Required columns are missing: {missing_columns}"
    )

# Validate data types and missing values
df["sales"] = pd.to_numeric(df["sales"], errors="raise")

if df["sales"].isna().any():
    raise ValueError("Sales column contains missing values.")

if (df["sales"] < 0).any():
    raise ValueError("Sales column contains negative values.")

print("\nDATA LOADED SUCCESSFULLY")
print("File:", file_path.name)

# ==========================================
# 3. GENERAL DATA SUMMARY
# ==========================================

print("\n" + "=" * 55)
print("1. DATA SUMMARY")
print("=" * 55)

print("Total daily records:", len(df))
print("Total columns:", len(df.columns))
print("Unique product-store series:", df["series_id"].nunique())
print("Unique products:", df["item_id"].nunique())
print("Unique stores:", df["store_id"].nunique())
print("Start date:", df["date"].min().date())
print("End date:", df["date"].max().date())

# ==========================================
# 4. OVERALL SALES SUMMARY
# ==========================================

print("\n" + "=" * 55)
print("2. OVERALL SALES SUMMARY")
print("=" * 55)

print("Total units sold:", int(df["sales"].sum()))
print("Average daily record sales:", round(df["sales"].mean(), 2))
print("Highest sales in one record:", int(df["sales"].max()))
print("Lowest sales in one record:", int(df["sales"].min()))
print("Median sales:", round(df["sales"].median(), 2))

print("\nDetailed statistical summary:")
print(df["sales"].describe().round(2))

# ==========================================
# 5. STORE-WISE SALES
# ==========================================

print("\n" + "=" * 55)
print("3. STORE-WISE SALES")
print("=" * 55)

store_sales = (
    df.groupby("store_id")["sales"]
    .sum()
    .sort_values(ascending=False)
)

print(store_sales.to_string())

# ==========================================
# 6. MONTH-WISE SALES
# ==========================================

print("\n" + "=" * 55)
print("4. MONTH-WISE SALES")
print("=" * 55)

monthly_sales = (
    df.groupby(df["date"].dt.to_period("M"))["sales"]
    .sum()
    .sort_index()
)

print("Latest 12 months in the dataset:")
print(monthly_sales.tail(12).to_string())

# ==========================================
# 7. TOP 10 PRODUCTS
# ==========================================

print("\n" + "=" * 55)
print("5. TOP 10 PRODUCTS BY TOTAL SALES")
print("=" * 55)

product_sales = (
    df.groupby("item_id")["sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print(product_sales.to_string())

# ==========================================
# 8. DAILY SALES SUMMARY
# ==========================================

print("\n" + "=" * 55)
print("6. DAILY SALES SUMMARY")
print("=" * 55)

daily_sales = (
    df.groupby("date")["sales"]
    .sum()
    .sort_index()
)

print("Average total sales per calendar day:",
      round(daily_sales.mean(), 2))

print("Highest total sales in one calendar day:",
      int(daily_sales.max()))

print("Lowest total sales in one calendar day:",
      int(daily_sales.min()))

# ==========================================
# 9. MISSING VALUES
# ==========================================

print("\n" + "=" * 55)
print("7. MISSING VALUE CHECK")
print("=" * 55)

missing_values = df.isnull().sum()
missing_values = missing_values[missing_values > 0]

if missing_values.empty:
    print("No missing values found.")
else:
    print(missing_values.to_string())

# ==========================================
# 10. SAVE ANALYSIS REPORTS
# ==========================================

reports_dir = BASE / "reports"
reports_dir.mkdir(parents=True, exist_ok=True)

store_report = reports_dir / "store_sales_summary.csv"
monthly_report = reports_dir / "monthly_sales_summary.csv"
product_report = reports_dir / "top_10_products.csv"
daily_report = reports_dir / "daily_sales_summary.csv"

store_sales.rename("total_sales").to_csv(store_report)

monthly_sales.rename("total_sales").to_csv(monthly_report)

product_sales.rename("total_sales").to_csv(product_report)

daily_sales.rename("total_sales").to_csv(daily_report)

print("\n" + "=" * 55)
print("8. REPORTS SAVED")
print("=" * 55)

print("Store report:", store_report)
print("Monthly report:", monthly_report)
print("Top products report:", product_report)
print("Daily report:", daily_report)

print("\nSales analysis completed successfully!")
print("Note: Results use the development sample of top 20 series.")
