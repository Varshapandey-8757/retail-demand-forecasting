
from datetime import date, datetime

from sqlalchemy import (
    Date,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    UniqueConstraint,
    func,
)
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
        Float, nullable=False
    )
    model_version: Mapped[str] = mapped_column(String(100), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (
        Index("ix_inventory_forecast_store_item", "store_id", "item_id"),
    )


class DailySales(Base):
    __tablename__ = "daily_sales"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    # Product and store information
    series_id: Mapped[str] = mapped_column(String(150), nullable=False)
    item_id: Mapped[str] = mapped_column(String(50), nullable=False)
    dept_id: Mapped[str] = mapped_column(String(50), nullable=False)
    cat_id: Mapped[str] = mapped_column(String(50), nullable=False)
    store_id: Mapped[str] = mapped_column(String(50), nullable=False)
    state_id: Mapped[str] = mapped_column(String(20), nullable=False)

    # Sales information
    d: Mapped[str] = mapped_column(String(20), nullable=False)
    sales: Mapped[int] = mapped_column(Integer, nullable=False)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    wm_yr_wk: Mapped[int] = mapped_column(Integer, nullable=False)
    sell_price: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Calendar information
    weekday: Mapped[str] = mapped_column(String(20), nullable=False)
    wday: Mapped[int] = mapped_column(Integer, nullable=False)
    month: Mapped[int] = mapped_column(Integer, nullable=False)
    year: Mapped[int] = mapped_column(Integer, nullable=False)

    # Event information
    event_name_1: Mapped[str | None] = mapped_column(String(100), nullable=True)
    event_type_1: Mapped[str | None] = mapped_column(String(100), nullable=True)
    event_name_2: Mapped[str | None] = mapped_column(String(100), nullable=True)
    event_type_2: Mapped[str | None] = mapped_column(String(100), nullable=True)

    # SNAP indicators
    snap_CA: Mapped[int] = mapped_column(Integer, nullable=False)
    snap_TX: Mapped[int] = mapped_column(Integer, nullable=False)
    snap_WI: Mapped[int] = mapped_column(Integer, nullable=False)

    __table_args__ = (
        UniqueConstraint("series_id", "d", name="uq_daily_sales_series_day"),
        Index("ix_daily_sales_date", "date"),
        Index("ix_daily_sales_store_item", "store_id", "item_id"),
    )
