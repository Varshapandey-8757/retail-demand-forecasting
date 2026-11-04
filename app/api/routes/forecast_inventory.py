from fastapi import APIRouter, HTTPException

from app.schemas.inventory_forecast import (
    InventoryForecastRequest,
    InventoryForecastResponse,
)

from app.services.inventory_forecast_service import (
    generate_inventory_recommendation,
)


router = APIRouter(
    prefix="/api/v1/inventory",
    tags=["Inventory Forecast"],
)


@router.post(
    "/forecast-recommendations",
    response_model=InventoryForecastResponse,
)
def forecast_inventory(request: InventoryForecastRequest):

    try:
        return generate_inventory_recommendation(request)

    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Trained ML model is not available.",
        )

    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        )