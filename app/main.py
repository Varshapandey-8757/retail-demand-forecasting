from fastapi import FastAPI
import fastapi
from app.api.routes.health import router as health_router
app = fastapi.FastAPI(title="Retail Demand Forecasting API",
    description="API for demand forecasting and inventory optimization",
    version="0.1.0")
app.include_router(health_router)