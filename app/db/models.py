from datetime import datetime

from sqlalchemy import DateTime, Float, Index, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base


class InventoryForecastResult(Base):
    __tablename__ = "inventory_forecast_results"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    store_id: Mapped[str] = mapped_column(String(50), nullable=False)
    item_id: Mapped[str] = mapped_column(String(50), nullable=False)

    predicted_units: Mapped[float] = mapped_column(Float, nullable=False)
    safety_stock: Mapped[float] = mapped_column(Float, nullable=False)
    current_stock: Mapped[float] = mapped_column(Float, nullable=False)
    incoming_stock: Mapped[float] = mapped_column(Float, nullable=False)

    recommended_order_quantity: Mapped[float] = mapped_column(
        Float,
        nullable=False
    )

    model_version: Mapped[str] = mapped_column(
        String(100),
        nullable=False
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )

    __table_args__ = (
        Index("ix_inventory_forecast_store_item", "store_id", "item_id"),
    )