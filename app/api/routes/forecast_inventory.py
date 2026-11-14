from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.db.database import get_db

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
def forecast_inventory(
    request: InventoryForecastRequest,
    db: Session = Depends(get_db),
):

    try:
        return generate_inventory_recommendation(request, db)

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

    except SQLAlchemyError:
        raise HTTPException(
            status_code=500,
            detail="Failed to save inventory forecast in database.",
        )