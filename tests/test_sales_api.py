
from datetime import date

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.db.database import Base, get_db
from app.db.models import DailySales


# In-memory SQLite database for testing
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

    db = TestingSessionLocal()

    sample_sales = [
        DailySales(
            series_id="FOODS_1_001_CA_1",
            item_id="FOODS_1_001",
            dept_id="FOODS_1",
            cat_id="FOODS",
            store_id="CA_1",
            state_id="CA",
            d="d_1",
            sales=10,
            date=date(2016, 1, 1),
            wm_yr_wk=11548,
            sell_price=2.50,
            weekday="Friday",
            wday=1,
            month=1,
            year=2016,
            snap_CA=0,
            snap_TX=0,
            snap_WI=0,
        ),
        DailySales(
            series_id="FOODS_1_001_CA_1",
            item_id="FOODS_1_001",
            dept_id="FOODS_1",
            cat_id="FOODS",
            store_id="CA_1",
            state_id="CA",
            d="d_2",
            sales=15,
            date=date(2016, 1, 2),
            wm_yr_wk=11548,
            sell_price=2.50,
            weekday="Saturday",
            wday=2,
            month=1,
            year=2016,
            snap_CA=0,
            snap_TX=0,
            snap_WI=0,
        ),
        DailySales(
            series_id="FOODS_1_002_TX_1",
            item_id="FOODS_1_002",
            dept_id="FOODS_1",
            cat_id="FOODS",
            store_id="TX_1",
            state_id="TX",
            d="d_1",
            sales=20,
            date=date(2016, 1, 1),
            wm_yr_wk=11548,
            sell_price=3.00,
            weekday="Friday",
            wday=1,
            month=1,
            year=2016,
            snap_CA=0,
            snap_TX=1,
            snap_WI=0,
        ),
    ]

    try:
        db.add_all(sample_sales)
        db.commit()
    finally:
        db.close()

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=test_engine)


def test_sales_endpoint_returns_records(client):
    response = client.get("/api/v1/sales")

    assert response.status_code == 200
    assert response.json()["total_records"] == 3
    assert len(response.json()["sales"]) == 3


def test_sales_store_filter(client):
    response = client.get(
        "/api/v1/sales",
        params={"store_id": "CA_1"},
    )

    assert response.status_code == 200
    assert response.json()["total_records"] == 2

    for record in response.json()["sales"]:
        assert record["store_id"] == "CA_1"


def test_sales_item_filter(client):
    response = client.get(
        "/api/v1/sales",
        params={"item_id": "FOODS_1_002"},
    )

    assert response.status_code == 200
    assert response.json()["total_records"] == 1
    assert response.json()["sales"][0]["item_id"] == "FOODS_1_002"


def test_sales_date_filter(client):
    response = client.get(
        "/api/v1/sales",
        params={
            "start_date": "2016-01-02",
            "end_date": "2016-01-02",
        },
    )

    assert response.status_code == 200
    assert response.json()["total_records"] == 1
    assert response.json()["sales"][0]["sales"] == 15


def test_sales_pagination(client):
    response = client.get(
        "/api/v1/sales",
        params={"limit": 1, "offset": 1},
    )

    assert response.status_code == 200
    assert response.json()["total_records"] == 3
    assert response.json()["limit"] == 1
    assert response.json()["offset"] == 1
    assert len(response.json()["sales"]) == 1


def test_sales_invalid_date_range(client):
    response = client.get(
        "/api/v1/sales",
        params={
            "start_date": "2016-01-10",
            "end_date": "2016-01-01",
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"] == (
        "start_date cannot be later than end_date"
    )
