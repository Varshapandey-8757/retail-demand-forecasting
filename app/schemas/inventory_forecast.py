from pydantic import BaseModel, Field


class InventoryForecastRequest(BaseModel):
    # Product identification
    store_id: str
    item_id: str

    # ML model features
    lag_1: float = Field(ge=0)
    lag_7: float = Field(ge=0)
    lag_28: float = Field(ge=0)
    rolling_mean_7: float = Field(ge=0)
    rolling_mean_28: float = Field(ge=0)
    day_of_week: int = Field(ge=0, le=6)
    month: int = Field(ge=1, le=12)
    day_of_month: int = Field(ge=1, le=31)

    # Inventory information
    safety_stock: float = Field(ge=0)
    current_stock: float = Field(ge=0)
    incoming_stock: float = Field(ge=0)


class InventoryForecastResponse(BaseModel):
    store_id: str
    item_id: str
    predicted_units: float
    safety_stock: float
    current_stock: float
    incoming_stock: float
    recommended_order_quantity: float
    model_version: str