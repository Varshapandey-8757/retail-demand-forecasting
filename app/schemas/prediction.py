from pydantic import BaseModel, Field


class DemandPredictionRequest(BaseModel):
    store_id: str
    item_id: str

    lag_1: float = Field(ge=0)
    lag_7: float = Field(ge=0)
    lag_28: float = Field(ge=0)

    rolling_mean_7: float = Field(ge=0)
    rolling_mean_28: float = Field(ge=0)

    day_of_week: int = Field(ge=0, le=6)
    month: int = Field(ge=1, le=12)
    day_of_month: int = Field(ge=1, le=31)


class DemandPredictionResponse(BaseModel):
    store_id: str
    item_id: str
    predicted_units: float
    model_version: str