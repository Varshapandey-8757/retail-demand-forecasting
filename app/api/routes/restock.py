from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.schemas.restock import RestockPlanRequest, RestockPlanResponse
from app.services.forecast_service import backtest_residual_std, get_forecasts
from app.services.restock_service import build_restock_plan, safety_stock_units

router = APIRouter(
    prefix="/api/v1/inventory",
    tags=["Restocking"],
)


@router.post("/restock-plan", response_model=RestockPlanResponse)
def restock_plan(request: RestockPlanRequest, db: Session = Depends(get_db)):
    """Restocking schedule for one store/item from the stored 30-day forecast."""
    run_id, rows = get_forecasts(
        db, model_name="lightgbm", group_by="series",
        store_id=request.store_id, item_id=request.item_id,
    )
    if not rows:
        raise HTTPException(
            status_code=404,
            detail="No stored forecast for this store/item. "
                   "Run: python -m pipelines.run_lightgbm_forecast",
        )

    dates = [r.forecast_date for r in rows]
    forecast = [r.predicted_units for r in rows]

    sigma = backtest_residual_std(db, request.store_id, request.item_id)
    if sigma is None:
        sigma = 0.3 * (sum(forecast) / len(forecast))
        basis = "fallback: 30% of mean forecast (no backtest residuals stored)"
    else:
        basis = "std-dev of backtest forecast errors for this item"
    safety = safety_stock_units(sigma, request.lead_time_days,
                                request.review_period_days, request.service_level)

    plan = build_restock_plan(
        dates=dates, forecast=forecast, current_stock=request.current_stock,
        incoming={s.days_from_now: s.quantity for s in request.incoming},
        lead_time_days=request.lead_time_days,
        review_period_days=request.review_period_days,
        safety_stock=safety,
    )
    return RestockPlanResponse(
        store_id=request.store_id, item_id=request.item_id,
        model_version=rows[0].model_version,
        safety_stock=round(safety, 2), safety_stock_basis=basis, **plan,
    )