from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.schemas.forecast import ForecastResponse, GroupBy, WhatIfRequest, WhatIfResponse
from app.services.forecast_service import get_forecasts
from app.services.whatif_service import run_what_if

router = APIRouter(
    prefix="/api/v1/forecasts",
    tags=["Forecasts"],
)


@router.get("", response_model=ForecastResponse)
def read_forecasts(
    model_name: str = Query(default="lightgbm", pattern="^(lightgbm|prophet)$"),
    group_by: GroupBy = Query(default="series", description="series, store, dept or category"),
    store_id: str | None = Query(default=None),
    item_id: str | None = Query(default=None),
    dept_id: str | None = Query(default=None),
    cat_id: str | None = Query(default=None),
    start_date: date | None = Query(default=None),
    end_date: date | None = Query(default=None),
    include_backtest: bool = Query(default=False, description="Also return past days with actuals"),
    limit: int = Query(default=10000, ge=1, le=100000),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    if start_date and end_date and start_date > end_date:
        raise HTTPException(status_code=422, detail="start_date cannot be later than end_date")

    run_id, forecasts = get_forecasts(
        db, model_name=model_name, group_by=group_by,
        store_id=store_id, item_id=item_id, dept_id=dept_id, cat_id=cat_id,
        start_date=start_date, end_date=end_date,
        include_backtest=include_backtest, limit=limit, offset=offset,
    )
    return ForecastResponse(
        source="demo" if run_id == "demo" else "database",
        model_name=model_name, run_id=run_id,
        group_by=group_by, forecasts=forecasts,
    )


@router.post("/what-if", response_model=WhatIfResponse)
def what_if(request: WhatIfRequest, db: Session = Depends(get_db)):
    try:
        return run_what_if(db, request)
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Trained LightGBM model is not available. "
                            "Run: python -m pipelines.run_lightgbm_forecast")
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error))
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))