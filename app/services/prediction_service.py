
from app.schemas.prediction import (
    DemandPredictionRequest,
    DemandPredictionResponse,
)

from app.services.model_service import predict_demand


def generate_prediction(
    request: DemandPredictionRequest,
) -> DemandPredictionResponse:

    features = {
        "lag_1": request.lag_1,
        "lag_7": request.lag_7,
        "lag_28": request.lag_28,
        "rolling_mean_7": request.rolling_mean_7,
        "rolling_mean_28": request.rolling_mean_28,
        "day_of_week": request.day_of_week,
        "month": request.month,
        "day_of_month": request.day_of_month,
    }

    prediction = predict_demand(features)

    return DemandPredictionResponse(
        store_id=request.store_id,
        item_id=request.item_id,
        predicted_units=round(prediction, 2),
        model_version="random-forest-v1",
    )
