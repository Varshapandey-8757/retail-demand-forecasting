from app.schemas.inventory_forecast import (
    InventoryForecastRequest,
    InventoryForecastResponse,
)

from app.schemas.prediction import DemandPredictionRequest
from app.services.prediction_service import generate_prediction
from app.services.inventory_service import calculate_reorder_quantity


def generate_inventory_recommendation(
    request: InventoryForecastRequest,
) -> InventoryForecastResponse:

    # Step 1: Prepare the ML prediction request
    prediction_request = DemandPredictionRequest(
        store_id=request.store_id,
        item_id=request.item_id,
        lag_1=request.lag_1,
        lag_7=request.lag_7,
        lag_28=request.lag_28,
        rolling_mean_7=request.rolling_mean_7,
        rolling_mean_28=request.rolling_mean_28,
        day_of_week=request.day_of_week,
        month=request.month,
        day_of_month=request.day_of_month,
    )

    # Step 2: Predict product demand
    prediction = generate_prediction(prediction_request)

    # Step 3: Calculate reorder quantity
    reorder_quantity = calculate_reorder_quantity(
        forecast_demand=prediction.predicted_units,
        safety_stock=request.safety_stock,
        current_stock=request.current_stock,
        incoming_stock=request.incoming_stock,
    )

    # Step 4: Return combined result
    return InventoryForecastResponse(
        store_id=request.store_id,
        item_id=request.item_id,
        predicted_units=prediction.predicted_units,
        safety_stock=request.safety_stock,
        current_stock=request.current_stock,
        incoming_stock=request.incoming_stock,
        recommended_order_quantity=round(reorder_quantity, 2),
        model_version=prediction.model_version,
    )