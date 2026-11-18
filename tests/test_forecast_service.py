from pathlib import Path
import pytest
from app.services.forecast_service import get_forecasts

BASE_DIR = Path(__file__).resolve().parents[1]
PROPHET_FILE = BASE_DIR / "reports" / "prophet_future_30d_forecasts.csv"


def test_forecast_service_loads_prophet_data():
    """Verify get_forecasts successfully retrieves Prophet forecasts."""
    assert PROPHET_FILE.exists(), f"Prophet file missing: {PROPHET_FILE}"
    forecasts = get_forecasts()
    assert len(forecasts) == 600, f"Expected 600 records, got {len(forecasts)}"
    assert forecasts[0].model_version == "prophet-v1"


def test_forecast_service_filter_by_store():
    """Verify store_id filter restricts results correctly."""
    target_store = "CA_1"
    forecasts = get_forecasts(store_id=target_store)
    assert len(forecasts) > 0
    assert all(f.store_id == target_store for f in forecasts)


def test_forecast_service_filter_by_item():
    """Verify item_id filter restricts results correctly."""
    target_item = "FOODS_3_090"
    forecasts = get_forecasts(item_id=target_item)
    assert len(forecasts) > 0
    assert all(f.item_id == target_item for f in forecasts)


def test_forecast_service_filter_by_store_and_item():
    """Verify store and item joint filter returns exactly 30 daily predictions."""
    target_store = "CA_1"
    target_item = "FOODS_3_090"
    forecasts = get_forecasts(store_id=target_store, item_id=target_item)
    assert len(forecasts) == 30
    assert all(
        f.store_id == target_store and f.item_id == target_item
        for f in forecasts
    )
