
from fastapi import FastAPI

from app.api.routes.health import router as health_router
from app.api.routes.forecasts import router as forecasts_router
from app.api.routes.inventory import router as inventory_router
from app.api.routes.predictions import router as predictions_router

app = FastAPI(
    title="Retail Demand Forecasting API",
    description=(
        "API for retail demand forecasting "
        "and inventory optimization"
    ),
    version="0.1.0",
)

app.include_router(health_router)
app.include_router(forecasts_router)
app.include_router(inventory_router)
app.include_router(predictions_router)
