"""30-day demand forecast, restocking plan and price what-if scenarios."""
import httpx
import pandas as pd
import streamlit as st

from dashboard_auth import API_BASE_URL, api_headers, require_login

st.set_page_config(page_title="Forecast & Scenarios", page_icon="📈", layout="wide")
require_login()

FORECAST_URL = f"{API_BASE_URL}/api/v1/forecasts"
WHAT_IF_URL = f"{FORECAST_URL}/what-if"
RESTOCK_URL = f"{API_BASE_URL}/api/v1/inventory/restock-plan"


# ------------------------------------------------------------------ API helpers
def _detail(response: httpx.Response) -> str:
    try:
        return str(response.json().get("detail", response.text))
    except ValueError:
        return response.text


def api_get(url: str, params: dict) -> dict:
    response = httpx.get(url, params=params, headers=api_headers(), timeout=60.0)
    if response.status_code >= 400:
        raise RuntimeError(_detail(response))
    return response.json()


def api_post(url: str, payload: dict) -> dict:
    response = httpx.post(url, json=payload, headers=api_headers(), timeout=120.0)
    if response.status_code >= 400:
        raise RuntimeError(_detail(response))
    return response.json()


@st.cache_data(ttl=300)
def fetch_forecasts(model_name: str, group_by: str, include_backtest: bool,
                    store_id: str, dept_id: str, cat_id: str) -> pd.DataFrame:
    params = {"model_name": model_name, "group_by": group_by,
              "include_backtest": include_backtest, "limit": 100000}
    for key, value in (("store_id", store_id), ("dept_id", dept_id), ("cat_id", cat_id)):
        if value != "All":
            params[key] = value
    data = api_get(FORECAST_URL, params)
    df = pd.DataFrame(data["forecasts"])
    if not df.empty:
        df["forecast_date"] = pd.to_datetime(df["forecast_date"])
    return df


@st.cache_data(ttl=300)
def fetch_options(model_name: str) -> dict:
    """Distinct stores, departments, categories and store/item pairs for the selectors."""
    out = {}
    for group, column in (("store", "store_id"), ("dept", "dept_id"), ("category", "cat_id")):
        data = api_get(FORECAST_URL, {"model_name": model_name, "group_by": group, "limit": 100000})
        out[column] = sorted({f[column] for f in data["forecasts"]})
    series = api_get(FORECAST_URL, {"model_name": model_name, "group_by": "series", "limit": 100000})
    out["pairs"] = sorted({(f["store_id"], f["item_id"]) for f in series["forecasts"]})
    return out


# ------------------------------------------------------------------ page header
st.title("📈 Demand Forecast & Scenarios")
st.write("30-day demand forecasts, restocking schedules and price what-if scenarios.")

model_name = st.sidebar.selectbox("Forecast model", ["lightgbm", "prophet"])
try:
    options = fetch_options(model_name)
except (RuntimeError, httpx.HTTPError) as error:
    st.error("Could not load forecasts from the API.")
    st.caption(f"Technical details: {error}")
    st.info("Run `python -m pipelines.run_lightgbm_forecast` (and the Prophet pipeline) first, "
            "then start the API: `python -m uvicorn app.main:app --reload`.")
    st.stop()

if not options["pairs"]:
    st.warning(f"No stored {model_name} forecasts yet. Run the forecasting pipeline first.")
    st.stop()

tab_forecast, tab_restock, tab_whatif = st.tabs(
    ["30-day forecast", "Restocking plan", "What-if price scenario"]
)

# ------------------------------------------------------------------ tab 1: forecast
with tab_forecast:
    c1, c2, c3, c4 = st.columns(4)
    group_label = c1.selectbox("Level", ["Store", "Department", "Category", "Item (store-item)"])
    store = c2.selectbox("Store", ["All"] + options["store_id"])
    dept = c3.selectbox("Department", ["All"] + options["dept_id"])
    cat = c4.selectbox("Category", ["All"] + options["cat_id"])
    group_by = {"Store": "store", "Department": "dept", "Category": "category",
                "Item (store-item)": "series"}[group_label]
    show_backtest = st.checkbox("Also show the last 30 days: predicted vs actual", value=True)

    try:
        df = fetch_forecasts(model_name, group_by, show_backtest, store, dept, cat)
    except (RuntimeError, httpx.HTTPError) as error:
        st.error(f"Could not load forecasts: {error}")
        st.stop()

    if df.empty:
        st.info("No forecast rows match these filters.")
    else:
        key_cols = [c for c in ("store_id", "item_id", "dept_id", "cat_id") if df[c].notna().any()]
        df["label"] = df[key_cols].astype(str).agg(" | ".join, axis=1) if key_cols else "All"

        future = df[~df["is_backtest"]]
        total = future.groupby("forecast_date")["predicted_units"].sum()
        m1, m2, m3 = st.columns(3)
        m1.metric("Forecast next 30 days (units)", f"{total.sum():,.0f}")
        m2.metric("Average per day", f"{total.mean():,.1f}")
        m3.metric("Peak day", f"{total.idxmax():%d %b}  ({total.max():,.0f})")

        wide = df.pivot_table(index="forecast_date", columns="label",
                              values="predicted_units", aggfunc="sum")
        st.subheader("Predicted demand")
        st.line_chart(wide)

        backtest = df[df["is_backtest"] & df["actual_units"].notna()]
        if show_backtest and not backtest.empty:
            daily = backtest.groupby("forecast_date")[["predicted_units", "actual_units"]].sum()
            wape = (daily["predicted_units"] - daily["actual_units"]).abs().sum() / daily["actual_units"].sum() * 100
            st.subheader("Backtest: predicted vs actual (past 30 days)")
            st.caption(f"Aggregated error (WAPE) for the selection: {wape:.1f}%")
            st.line_chart(daily.rename(columns={"predicted_units": "Predicted", "actual_units": "Actual"}))

        shown = df.drop(columns=["label"]).sort_values("forecast_date")
        st.dataframe(shown, hide_index=True)
        st.download_button("Download CSV", shown.to_csv(index=False).encode(),
                           file_name=f"forecast_{model_name}_{group_by}.csv", mime="text/csv")

# ------------------------------------------------------------------ pick one item
def pick_item(key: str) -> tuple[str, str]:
    left, right = st.columns(2)
    store_id = left.selectbox("Store", sorted({s for s, _ in options["pairs"]}), key=f"{key}_store")
    items = sorted({i for s, i in options["pairs"] if s == store_id})
    item_id = right.selectbox("Item", items, key=f"{key}_item")
    return store_id, item_id


# ------------------------------------------------------------------ tab 2: restock
with tab_restock:
    st.write("Order schedule built from the stored 30-day forecast "
             "(periodic review, order-up-to level, safety stock from backtest errors).")
    store_id, item_id = pick_item("restock")
    a, b, c, d = st.columns(4)
    stock = a.number_input("Current stock (units)", min_value=0.0, value=100.0, step=10.0)
    lead = b.number_input("Lead time (days)", min_value=0, max_value=30, value=2)
    review = c.number_input("Review period (days)", min_value=1, max_value=30, value=7)
    service = d.slider("Service level", 0.80, 0.99, 0.95, 0.01)
    arriving_qty = st.number_input("Shipment already on the way (units)", min_value=0.0, value=0.0, step=10.0)
    arriving_day = st.number_input("...arriving in (days)", min_value=0, max_value=60, value=1)

    if st.button("Build restocking plan"):
        payload = {"store_id": store_id, "item_id": item_id, "current_stock": stock,
                   "lead_time_days": int(lead), "review_period_days": int(review),
                   "service_level": float(service),
                   "incoming": [{"days_from_now": int(arriving_day), "quantity": arriving_qty}]
                   if arriving_qty > 0 else []}
        try:
            plan = api_post(RESTOCK_URL, payload)
        except (RuntimeError, httpx.HTTPError) as error:
            st.error(str(error))
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Safety stock", f"{plan['safety_stock']:,.0f}")
            m2.metric("Total to order (30 days)", f"{plan['total_ordered_units']:,}")
            m3.metric("Projected shortage", f"{plan['projected_shortage_units']:,.0f} units")
            st.caption(f"Safety stock basis: {plan['safety_stock_basis']}")
            if plan["orders"]:
                st.subheader("Orders to place")
                st.dataframe(pd.DataFrame(plan["orders"]), hide_index=True)
            else:
                st.success("No order needed in the next 30 days.")
            sched = pd.DataFrame(plan["schedule"]).set_index("date")
            st.subheader("Projected stock")
            st.line_chart(sched[["end_of_day_stock", "forecast_units"]])

# ------------------------------------------------------------------ tab 3: what-if
with tab_whatif:
    st.write("Simulate a price change and see the effect on predicted demand and revenue. "
             "Uses the LightGBM model, which learns price sensitivity from past price changes.")
    store_id, item_id = pick_item("whatif")
    change = st.slider("Price change (%)", -30, 30, -10, 1)

    if st.button("Run scenario"):
        try:
            result = api_post(WHAT_IF_URL, {"store_id": store_id, "item_id": item_id,
                                            "price_change_pct": float(change), "horizon_days": 30})
        except (RuntimeError, httpx.HTTPError) as error:
            st.error(str(error))
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Price", f"${result['scenario_price']:.2f}",
                      f"{change:+d}% vs ${result['baseline_price']:.2f}")
            units_pct = result["units_change_pct"]
            m2.metric("Units (30 days)", f"{result['total_scenario_units']:,.0f}",
                      None if units_pct is None else f"{units_pct:+.1f}%")
            rev_pct = result["revenue_change_pct"]
            m3.metric("Revenue (30 days)", f"${result['scenario_revenue']:,.0f}",
                      None if rev_pct is None else f"{rev_pct:+.1f}%")
            if result["warning"]:
                st.warning(result["warning"])
            daily = pd.DataFrame(result["daily"]).set_index("date")
            st.line_chart(daily[["baseline_units", "scenario_units"]]
                          .rename(columns={"baseline_units": "Baseline", "scenario_units": "Scenario"}))
            st.caption(f"Model: {result['model_version']}. Scenario results are model estimates, "
                       "not guarantees.")