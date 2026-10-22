
from fastapi import APIRouter, Query

from app.schemas.forecast import ForecastResponse
from app.services.forecast_service import get_forecasts

router = APIRouter(
    prefix="/api/v1/forecasts",
    tags=["Forecasts"],
)


@router.get("", response_model=ForecastResponse)
def read_forecasts(
    store_id: str | None = Query(default=None),
    item_id: str | None = Query(default=None),
):
    forecasts = get_forecasts(
        store_id=store_id,
        item_id=item_id,
    )

    return ForecastResponse(
        source="demo",
        forecasts=forecasts,
    )
