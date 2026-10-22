
def calculate_reorder_quantity(
    forecast_demand: float,
    safety_stock: float,
    current_stock: float,
    incoming_stock: float,
) -> float:
    values = [
        forecast_demand,
        safety_stock,
        current_stock,
        incoming_stock,
    ]

    if any(value < 0 for value in values):
        raise ValueError("Inventory values cannot be negative.")

    reorder_quantity = (
        forecast_demand
        + safety_stock
        - current_stock
        - incoming_stock
    )

    return max(0.0, reorder_quantity)
