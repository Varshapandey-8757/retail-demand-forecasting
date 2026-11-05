
from fastapi import APIRouter, HTTPException

from app.schemas.prediction import (
    DemandPredictionRequest,
    DemandPredictionResponse,
)

from app.services.prediction_service import generate_prediction


router = APIRouter(
    prefix="/api/v1/predictions",
    tags=["ML Predictions"],
)


@router.post(
    "/predict",
    response_model=DemandPredictionResponse,
)
def predict(request: DemandPredictionRequest):

    try:
        return generate_prediction(request)

    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Trained ML model is not available.",
        )

    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        )
