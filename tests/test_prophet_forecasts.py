import json
from pathlib import Path
import pandas as pd
import pytest

BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = BASE_DIR / "configs" / "prophet_config.json"
FORECAST_FILE = BASE_DIR / "reports" / "prophet_future_30d_forecasts.csv"


@pytest.fixture(scope="module")
def config():
    assert CONFIG_PATH.exists(), f"Config missing at {CONFIG_PATH}"
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def forecasts():
    assert FORECAST_FILE.exists(), (
        f"Forecast output missing: {FORECAST_FILE}\n"
        "Run 'python pipelines/generate_prophet_forecasts.py' first."
    )
    return pd.read_csv(FORECAST_FILE)


def test_forecast_columns_match_schema(forecasts, config):
    """Verify generated columns match the team's shared ForecastRecord schema."""
    expected_cols = config["output_schema"]
    assert list(forecasts.columns) == expected_cols, (
        f"Columns mismatch! Expected: {expected_cols}, Found: {list(forecasts.columns)}"
    )


def test_no_null_values(forecasts):
    """Verify forecasts contain zero null/missing values."""
    null_counts = forecasts.isnull().sum().to_dict()
    assert forecasts.isnull().sum().sum() == 0, f"Found nulls: {null_counts}"


def test_non_negative_predictions(forecasts):
    """Verify sales predictions are strictly non-negative (>= 0)."""
    assert (forecasts["predicted_units"] >= 0).all(), (
        "Found negative forecast values!"
    )


def test_expected_record_count(forecasts, config):
    """Verify total rows equals series_count * 30 days (20 * 30 = 600)."""
    expected_count = (
        config["series_count"] * config["future_forecast"]["horizon_days"]
    )
    assert len(forecasts) == expected_count, (
        f"Expected {expected_count} records, but got {len(forecasts)}"
    )


def test_model_version_tagged(forecasts, config):
    """Verify model_version matches config tagging."""
    expected_version = config["model_version"]
    assert (forecasts["model_version"] == expected_version).all()