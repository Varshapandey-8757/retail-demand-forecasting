from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.db.models import InventoryForecastResult
from app.schemas.inventory_forecast import InventoryForecastResponse


def save_inventory_forecast(
    db: Session,
    result: InventoryForecastResponse,
) -> InventoryForecastResult:

    db_record = InventoryForecastResult(
        store_id=result.store_id,
        item_id=result.item_id,
        predicted_units=result.predicted_units,
        safety_stock=result.safety_stock,
        current_stock=result.current_stock,
        incoming_stock=result.incoming_stock,
        recommended_order_quantity=result.recommended_order_quantity,
        model_version=result.model_version,
    )

    try:
        db.add(db_record)
        db.commit()
        db.refresh(db_record)

        return db_record

    except SQLAlchemyError:
        db.rollback()
        raise