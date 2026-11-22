"""Small M5-shaped dataset with a known price effect, used by the tests."""
import numpy as np
import pandas as pd

WEEKDAY = np.array([1.0, 0.9, 0.9, 1.0, 1.2, 1.4, 1.3])  # Mon..Sun
ELASTICITY = -2.0  # sales ~ price ** -2


def make_synthetic_sales(days: int = 520, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-01", periods=days, freq="D")
    specs = [
        ("FOODS_1_001", "FOODS_1", "FOODS", "CA_1", "CA", 30.0, 2.50),
        ("FOODS_1_002", "FOODS_1", "FOODS", "TX_1", "TX", 18.0, 3.00),
        ("HOBBIES_1_001", "HOBBIES_1", "HOBBIES", "CA_1", "CA", 12.0, 5.00),
        ("HOUSEHOLD_1_001", "HOUSEHOLD_1", "HOUSEHOLD", "WI_1", "WI", 9.0, 4.00),
    ]
    event_days = set(rng.choice(days, size=12, replace=False).tolist())
    frames = []
    for item, dept, cat, store, state, base, price in specs:
        week_no = (np.arange(days) // 7)
        promo = (week_no % 4 == 3)  # every 4th week: 15% off
        prices = np.where(promo, price * 0.85, price)
        mean = base * WEEKDAY[dates.dayofweek] * (prices / price) ** ELASTICITY
        sales = rng.poisson(mean)
        ev = np.array([i in event_days for i in range(days)])
        frames.append(pd.DataFrame({
            "series_id": f"{item}_{store}_validation",
            "item_id": item, "dept_id": dept, "cat_id": cat,
            "store_id": store, "state_id": state,
            "date": dates, "sales": sales, "sell_price": prices,
            "event_name_1": np.where(ev, "SuperBowl", None),
            "event_type_1": np.where(ev, "Sporting", None),
            "event_name_2": None, "event_type_2": None,
            "snap_CA": (dates.day <= 10).astype(int),
            "snap_TX": (dates.day % 2 == 0).astype(int),
            "snap_WI": 0,
        }))
    return pd.concat(frames, ignore_index=True)