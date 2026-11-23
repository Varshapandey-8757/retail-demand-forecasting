from fastapi import Depends, FastAPI

from app.api.routes.health import router as health_router
from app.api.routes.forecasts import router as forecasts_router
from app.api.routes.inventory import router as inventory_router
from app.api.routes.predictions import router as predictions_router
from app.api.routes.forecast_inventory import router as forecast_inventory_router
from app.api.routes.restock import router as restock_router
from app.api.routes.sales import router as sales_router
from app.security import require_admin, require_viewer
from app.db.database import Base, engine

app = FastAPI(
    title="Retail Demand Forecasting API",
    description=(
        "API for retail demand forecasting "
        "and inventory optimization"
    ),
    version="0.2.0",
)


@app.on_event("startup")
def create_database_tables() -> None:
    """Create local SQLite tables automatically when no warehouse is configured."""
    Base.metadata.create_all(bind=engine)

# Public: health check only.
app.include_router(health_router)

# Read access (viewer or admin API key).
for router in (
    forecasts_router,
    inventory_router,
    predictions_router,
    restock_router,
    sales_router,
):
    app.include_router(router, dependencies=[Depends(require_viewer)])

# Writes to the database (admin API key).
app.include_router(forecast_inventory_router, dependencies=[Depends(require_admin)])