
from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import DailySales
from app.schemas.sales import SalesResponse

router = APIRouter(
    prefix="/api/v1/sales",
    tags=["Sales"],
)


@router.get("", response_model=SalesResponse)
def read_sales(
    store_id: str | None = Query(default=None),
    item_id: str | None = Query(default=None),
    start_date: date | None = Query(default=None),
    end_date: date | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    if start_date and end_date and start_date > end_date:
        raise HTTPException(
            status_code=422,
            detail="start_date cannot be later than end_date",
        )

    query = db.query(DailySales)

    if store_id:
        query = query.filter(DailySales.store_id == store_id)

    if item_id:
        query = query.filter(DailySales.item_id == item_id)

    if start_date:
        query = query.filter(DailySales.date >= start_date)

    if end_date:
        query = query.filter(DailySales.date <= end_date)

    total_records = query.count()

    sales_records = (
        query
        .order_by(DailySales.date.asc(), DailySales.id.asc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    return SalesResponse(
        total_records=total_records,
        limit=limit,
        offset=offset,
        sales=sales_records,
    )
