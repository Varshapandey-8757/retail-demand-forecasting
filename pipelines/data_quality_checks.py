"""Data quality checks for the daily sales data (Week 1: ETL quality).

Checks the table/CSV before modelling and writes reports/data_quality_report.json.
Exit code is 1 if any ERROR is found, so it can be used in CI or before training.

    python -m pipelines.data_quality_checks                 # reads PostgreSQL daily_sales
    python -m pipelines.data_quality_checks --csv data/processed/retail_sales_daily_top20.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = [
    "series_id", "item_id", "dept_id", "cat_id", "store_id", "state_id",
    "date", "sales", "sell_price", "snap_CA", "snap_TX", "snap_WI",
]
KEY_COLUMNS = ["series_id", "item_id", "dept_id", "cat_id", "store_id", "date", "sales"]
ALLOWED = {"cat_id": {"FOODS", "HOBBIES", "HOUSEHOLD"}, "state_id": {"CA", "TX", "WI"}}
REPORT_PATH = Path(__file__).resolve().parents[1] / "reports" / "data_quality_report.json"


def _issue(name: str, severity: str, count: int, detail: str) -> dict:
    return {"check": name, "severity": severity, "count": int(count), "detail": detail}


def run_checks(df: pd.DataFrame) -> dict:
    issues: list[dict] = []

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        issues.append(_issue("required_columns", "error", len(missing), f"Missing: {missing}"))
        return _report(df, issues)  # nothing else can run safely

    d = df.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d["sales"] = pd.to_numeric(d["sales"], errors="coerce")
    d["sell_price"] = pd.to_numeric(d["sell_price"], errors="coerce")

    # --- nulls / bad types in key columns
    for col in KEY_COLUMNS:
        n = int(d[col].isna().sum())
        if n:
            issues.append(_issue("null_key_column", "error", n, f"{col} has {n} null/invalid values"))

    # --- sales values
    n = int((d["sales"] < 0).sum())
    if n:
        issues.append(_issue("negative_sales", "error", n, "sales < 0"))
    n = int(((d["sales"] % 1) != 0).sum())
    if n:
        issues.append(_issue("fractional_sales", "error", n, "sales must be whole units"))

    # --- prices
    n = int((d["sell_price"] <= 0).sum())
    if n:
        issues.append(_issue("non_positive_price", "error", n, "sell_price <= 0"))
    n = int((d["sell_price"].isna() & (d["sales"] > 0)).sum())
    if n:
        issues.append(_issue("sold_without_price", "error", n, "units sold on days with no price"))
    share = float(d["sell_price"].isna().mean() * 100)
    if share > 0:
        issues.append(_issue("null_price_share", "warning", int(d["sell_price"].isna().sum()),
                             f"{share:.1f}% rows have no price (normal before an item is first sold)"))

    # --- duplicates and gaps
    n = int(d.duplicated(["series_id", "date"]).sum())
    if n:
        issues.append(_issue("duplicate_series_day", "error", n, "more than one row per series and day"))
    d = d.dropna(subset=["date"]).drop_duplicates(["series_id", "date"]).sort_values(["series_id", "date"])
    gaps = d.groupby("series_id")["date"].diff().dropna()
    n = int((gaps != pd.Timedelta(days=1)).sum())
    if n:
        issues.append(_issue("date_gaps", "error", n,
                             "series have missing days (lag features and Prophet need daily data)"))

    # --- categorical / flag values
    for col, allowed in ALLOWED.items():
        n = int((~d[col].isin(allowed) & d[col].notna()).sum())
        if n:
            issues.append(_issue("unexpected_category", "error", n, f"{col} not in {sorted(allowed)}"))
    for col in ("snap_CA", "snap_TX", "snap_WI"):
        n = int((~d[col].isin([0, 1]) & d[col].notna()).sum())
        if n:
            issues.append(_issue("invalid_snap_flag", "error", n, f"{col} must be 0 or 1"))

    # --- softer warnings
    n = int((d["date"] > pd.Timestamp.today().normalize()).sum())
    if n:
        issues.append(_issue("future_dates", "warning", n, "rows dated after today"))
    zero = d.groupby("series_id")["sales"].sum()
    n = int((zero == 0).sum())
    if n:
        issues.append(_issue("all_zero_series", "warning", n, "series with no sales at all"))
    stats = d.groupby("series_id")["sales"].agg(["mean", "std"])
    joined = d.join(stats, on="series_id")
    n = int((joined["sales"] > joined["mean"] + 10 * joined["std"].fillna(0)).sum())
    if n:
        issues.append(_issue("extreme_outliers", "warning", n, "sales above mean + 10 std of their series"))

    return _report(d, issues)


def _report(df: pd.DataFrame, issues: list[dict]) -> dict:
    errors = [i for i in issues if i["severity"] == "error"]
    summary = {"rows": int(len(df))}
    if {"series_id", "date"}.issubset(df.columns) and len(df):
        dates = pd.to_datetime(df["date"], errors="coerce")
        summary.update(series=int(df["series_id"].nunique()),
                       first_date=str(dates.min().date()), last_date=str(dates.max().date()))
    return {"passed": not errors, "errors": len(errors),
            "warnings": len(issues) - len(errors), "summary": summary, "issues": issues}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", help="Check this CSV instead of the PostgreSQL daily_sales table")
    ap.add_argument("--report", default=str(REPORT_PATH))
    args = ap.parse_args()

    if args.csv:
        df = pd.read_csv(args.csv)
    else:
        from app.ml.data import get_engine
        df = pd.read_sql_table("daily_sales", get_engine())

    report = run_checks(df)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2))

    print(f"Rows: {report['summary']['rows']:,} | errors: {report['errors']} | warnings: {report['warnings']}")
    for issue in report["issues"]:
        print(f"  [{issue['severity'].upper()}] {issue['check']}: {issue['detail']} ({issue['count']})")
    print("PASSED" if report["passed"] else "FAILED")
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()