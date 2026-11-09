import pytest

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.db.database import Base, get_db
from app.db.models import InventoryForecastResult


# Create an in-memory SQLite database for testing
TEST_DATABASE_URL = "sqlite://"

test_engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)

TestingSessionLocal = sessionmaker(
    bind=test_engine,
    autoflush=False,
    autocommit=False,
)


@pytest.fixture
def client():
    Base.metadata.create_all(bind=test_engine)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=test_engine)


def test_inventory_forecast_is_saved_in_database(client):

    payload = {
        "store_id": "CA_1",
        "item_id": "FOODS_1_001",
        "lag_1": 5,
        "lag_7": 6,
        "lag_28": 4,
        "rolling_mean_7": 5.5,
        "rolling_mean_28": 4.8,
        "day_of_week": 2,
        "month": 3,
        "day_of_month": 15,
        "safety_stock": 10,
        "current_stock": 20,
        "incoming_stock": 5,
    }

    response = client.post(
        "/api/v1/inventory/forecast-recommendations",
        json=payload,
    )

    assert response.status_code == 200

    db = TestingSessionLocal()

    try:
        saved_record = (
            db.query(InventoryForecastResult)
            .filter_by(
                store_id="CA_1",
                item_id="FOODS_1_001",
            )
            .first()
        )

        assert saved_record is not None
        assert saved_record.predicted_units >= 0
        assert saved_record.recommended_order_quantity >= 0
        assert saved_record.model_version == "random-forest-v1"

    finally:
        db.close()