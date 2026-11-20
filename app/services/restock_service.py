"""Periodic-review restocking plan driven by the demand forecast.

Every `review_period_days` we look at the inventory position (stock on hand +
shipments already on the way) and order up to
    forecast demand over (lead time + review period) + safety stock.
Orders arrive `lead_time_days` later. Unmet demand is treated as lost sales.
"""
from __future__ import annotations

import math
from collections import defaultdict
from datetime import date, timedelta
from statistics import NormalDist


def safety_stock_units(sigma_daily: float, lead_time_days: int,
                       review_period_days: int, service_level: float) -> float:
    """z * sigma * sqrt(protection interval) - the classic normal-demand formula."""
    z = NormalDist().inv_cdf(service_level)
    return max(0.0, z * sigma_daily * math.sqrt(lead_time_days + review_period_days))


def build_restock_plan(
    dates: list[date],
    forecast: list[float],
    current_stock: float,
    incoming: dict[int, float],
    lead_time_days: int,
    review_period_days: int,
    safety_stock: float,
) -> dict:
    if len(dates) != len(forecast) or not dates:
        raise ValueError("dates and forecast must be non-empty and the same length")

    horizon = len(dates)
    arrivals: dict[int, float] = defaultdict(float, incoming)
    stock = float(current_stock)
    shortage = 0.0
    orders, schedule = [], []

    for day in range(horizon):
        stock += arrivals.get(day, 0.0)

        if day % review_period_days == 0:
            on_order = sum(q for d, q in arrivals.items() if d > day)
            position = stock + on_order
            cover = sum(forecast[day: day + lead_time_days + review_period_days])
            order_qty = math.ceil(max(0.0, cover + safety_stock - position))
            if order_qty > 0:
                arrivals[day + lead_time_days] += order_qty
                orders.append({
                    "order_date": dates[day],
                    "arrival_date": dates[day] + timedelta(days=lead_time_days),
                    "quantity": order_qty,
                })

        demand = forecast[day]
        sold = min(stock, demand)
        shortage += demand - sold
        stock -= sold
        schedule.append({"date": dates[day], "forecast_units": round(demand, 2),
                         "end_of_day_stock": round(stock, 2)})

    return {
        "orders": orders,
        "total_ordered_units": int(sum(o["quantity"] for o in orders)),
        "projected_shortage_units": round(shortage, 2),
        "schedule": schedule,
    }