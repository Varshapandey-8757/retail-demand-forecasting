
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_forecasts_endpoint():
    response = client.get("/api/v1/forecasts")

    assert response.status_code == 200

    data = response.json()
    assert data["source"] == "demo"
    assert len(data["forecasts"]) > 0


def test_forecast_store_filter():
    response = client.get(
        "/api/v1/forecasts",
        params={"store_id": "STORE_001"},
    )

    assert response.status_code == 200

    for record in response.json()["forecasts"]:
        assert record["store_id"] == "STORE_001"


def test_inventory_recommendation():
    response = client.post(
        "/api/v1/inventory/recommendations",
        json={
            "forecast_demand": 100,
            "safety_stock": 20,
            "current_stock": 30,
            "incoming_stock": 10,
        },
    )

    assert response.status_code == 200
    assert response.json()["recommended_order_quantity"] == 80


def test_inventory_rejects_negative_values():
    response = client.post(
        "/api/v1/inventory/recommendations",
        json={
            "forecast_demand": -5,
            "safety_stock": 20,
            "current_stock": 30,
            "incoming_stock": 10,
        },
    )

    assert response.status_code == 422
