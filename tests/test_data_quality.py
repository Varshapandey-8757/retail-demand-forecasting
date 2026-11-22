import pandas as pd

from pipelines.data_quality_checks import run_checks
from tests.synthetic_data import make_synthetic_sales


def names(report, severity="error"):
    return {i["check"] for i in report["issues"] if i["severity"] == severity}


def test_clean_data_passes():
    report = run_checks(make_synthetic_sales())
    assert report["passed"], report["issues"]
    assert report["summary"]["series"] == 4


def test_negative_sales_and_bad_price_are_errors():
    df = make_synthetic_sales()
    df.loc[5, "sales"] = -3
    df.loc[6, "sell_price"] = 0
    report = run_checks(df)
    assert not report["passed"]
    assert {"negative_sales", "non_positive_price"} <= names(report)


def test_duplicates_and_gaps_are_errors():
    df = make_synthetic_sales()
    dup = pd.concat([df, df.iloc[[10]]], ignore_index=True)
    assert "duplicate_series_day" in names(run_checks(dup))
    gap = df.drop(df.index[20])
    assert "date_gaps" in names(run_checks(gap))


def test_sold_without_price_is_error_but_launch_gap_is_only_warning():
    df = make_synthetic_sales()
    df.loc[0:2, "sell_price"] = None
    df.loc[0:2, "sales"] = 0           # not sold yet -> fine
    report = run_checks(df)
    assert report["passed"]
    assert "null_price_share" in names(report, "warning")
    df.loc[0, "sales"] = 4             # sold with no price -> error
    assert "sold_without_price" in names(run_checks(df))


def test_unexpected_category_and_missing_columns():
    df = make_synthetic_sales()
    df.loc[0, "cat_id"] = "TOYS"
    assert "unexpected_category" in names(run_checks(df))
    assert "required_columns" in names(run_checks(df.drop(columns=["sell_price"])))