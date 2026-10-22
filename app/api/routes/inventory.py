
from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.services.inventory_service import calculate_reorder_quantity

router = APIRouter(
    prefix="/api/v1/inventory",
    tags=["Inventory"],
)


class InventoryRequest(BaseModel):
    forecast_demand: float = Field(ge=0)
    safety_stock: float = Field(ge=0)
    current_stock: float = Field(ge=0)
    incoming_stock: float = Field(ge=0)


class InventoryResponse(BaseModel):
    recommended_order_quantity: float


@router.post(
    "/recommendations",
    response_model=InventoryResponse,
)
def inventory_recommendation(
    request: InventoryRequest,
):
    quantity = calculate_reorder_quantity(
        forecast_demand=request.forecast_demand,
        safety_stock=request.safety_stock,
        current_stock=request.current_stock,
        incoming_stock=request.incoming_stock,
    )

    return InventoryResponse(
        recommended_order_quantity=quantity
    )
