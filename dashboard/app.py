
import streamlit as st
import pandas as pd
import httpx


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Retail Demand Forecasting Dashboard",
    page_icon="📊",
    layout="wide"
)


# --------------------------------------------------
# API CONFIGURATION
# --------------------------------------------------

API_BASE_URL = "http://127.0.0.1:8000"
SALES_API_URL = f"{API_BASE_URL}/api/v1/sales"


# --------------------------------------------------
# PAGE TITLE
# --------------------------------------------------

st.title("📊 Retail Demand Forecasting & Inventory Optimization")

st.write(
    "Analyze historical retail sales data and understand "
    "product and store performance."
)

st.divider()


# --------------------------------------------------
# FETCH SALES DATA FROM FASTAPI
# --------------------------------------------------

@st.cache_data(ttl=300)
def fetch_sales_data():

    all_sales = []
    limit = 1000
    offset = 0

    try:

        with httpx.Client(timeout=30.0) as client:

            while True:

                response = client.get(
                    SALES_API_URL,
                    params={
                        "limit": limit,
                        "offset": offset
                    }
                )

                response.raise_for_status()

                result = response.json()

                batch = result.get("sales", [])

                all_sales.extend(batch)

                total_records = result.get(
                    "total_records", 0
                )

                if not batch:
                    break

                offset += len(batch)

                if offset >= total_records:
                    break

        df = pd.DataFrame(all_sales)

        if not df.empty and "date" in df.columns:
            df["date"] = pd.to_datetime(
                df["date"],
                errors="coerce"
            )

        return df, None

    except httpx.HTTPError as error:

        return pd.DataFrame(), str(error)

    except (ValueError, TypeError) as error:

        return pd.DataFrame(), str(error)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

with st.spinner("Loading sales data from PostgreSQL API..."):

    df, error = fetch_sales_data()


if error:

    st.error(
        "Unable to connect to the Sales API."
    )

    st.write("Please check the following:")

    st.write("- FastAPI server is running.")
    st.write("- PostgreSQL connection is working.")
    st.write("- Sales API endpoint is available.")

    st.code("python -m uvicorn app.main:app --reload")

    st.caption(f"Technical details: {error}")

    st.stop()


if df.empty:

    st.warning("No sales records found.")

    st.stop()


# --------------------------------------------------
# SIDEBAR FILTERS
# --------------------------------------------------

st.sidebar.header("Dashboard Filters")

filtered_df = df.copy()


# Date Filter

if "date" in filtered_df.columns:

    min_date = filtered_df["date"].min().date()
    max_date = filtered_df["date"].max().date()

    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    if isinstance(date_range, tuple) and len(date_range) == 2:

        start_date, end_date = date_range

        filtered_df = filtered_df[
            (filtered_df["date"].dt.date >= start_date)
            &
            (filtered_df["date"].dt.date <= end_date)
        ]


# Store Filter

if "store_id" in filtered_df.columns:

    stores = sorted(
        filtered_df["store_id"].dropna().unique().tolist()
    )

    selected_stores = st.sidebar.multiselect(
        "Select Store",
        options=stores,
        default=stores
    )

    filtered_df = filtered_df[
        filtered_df["store_id"].isin(selected_stores)
    ]


# Product Filter

if "item_id" in filtered_df.columns:

    products = sorted(
        filtered_df["item_id"].dropna().unique().tolist()
    )

    selected_products = st.sidebar.multiselect(
        "Select Product",
        options=products,
        default=products
    )

    filtered_df = filtered_df[
        filtered_df["item_id"].isin(selected_products)
    ]


st.sidebar.divider()

st.sidebar.write(
    f"Loaded records: {len(df):,}"
)

st.sidebar.write(
    f"Filtered records: {len(filtered_df):,}"
)


# --------------------------------------------------
# EXECUTIVE OVERVIEW
# --------------------------------------------------

st.header("Executive Overview")

if filtered_df.empty:

    st.warning("No records match the selected filters.")

    st.stop()


total_records = len(filtered_df)

total_sales = filtered_df["sales"].sum()

total_stores = (
    filtered_df["store_id"].nunique()
    if "store_id" in filtered_df.columns
    else 0
)

total_products = (
    filtered_df["item_id"].nunique()
    if "item_id" in filtered_df.columns
    else 0
)


col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Sales Records",
    f"{total_records:,}"
)

col2.metric(
    "Total Units Sold",
    f"{total_sales:,.0f}"
)

col3.metric(
    "Total Stores",
    f"{total_stores:,}"
)

col4.metric(
    "Unique Products",
    f"{total_products:,}"
)


st.divider()


# --------------------------------------------------
# DAILY SALES TREND
# --------------------------------------------------

st.header("Daily Sales Trend")

if "date" in filtered_df.columns:

    daily_sales = (
        filtered_df
        .groupby("date")["sales"]
        .sum()
        .reset_index()
        .sort_values("date")
    )

    daily_sales = daily_sales.set_index("date")

    st.line_chart(
        daily_sales,
        y="sales",
        use_container_width=True
    )


st.divider()


# --------------------------------------------------
# TOP 10 PRODUCTS
# --------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    st.subheader("Top 10 Products")

    if "item_id" in filtered_df.columns:

        top_products = (
            filtered_df
            .groupby("item_id")["sales"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )

        st.bar_chart(top_products)


# --------------------------------------------------
# STORE-WISE SALES
# --------------------------------------------------

with col2:

    st.subheader("Store-wise Sales")

    if "store_id" in filtered_df.columns:

        store_sales = (
            filtered_df
            .groupby("store_id")["sales"]
            .sum()
            .sort_values(ascending=False)
        )

        st.bar_chart(store_sales)


st.divider()


# --------------------------------------------------
# HISTORICAL SALES TABLE
# --------------------------------------------------

st.header("Historical Sales Records")

st.write(
    "Explore the historical retail sales records "
    "retrieved from PostgreSQL."
)

display_columns = [
    column for column in [
        "date",
        "series_id",
        "store_id",
        "item_id",
        "sales",
        "sell_price"
    ]
    if column in filtered_df.columns
]

st.dataframe(
    filtered_df[display_columns].sort_values(
        by="date",
        ascending=False
    ),
    use_container_width=True,
    hide_index=True
)


# --------------------------------------------------
# DOWNLOAD FILTERED DATA
# --------------------------------------------------

csv_data = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Filtered Sales Data",
    data=csv_data,
    file_name="retail_sales_analysis.csv",
    mime="text/csv"
)


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Retail Demand Forecasting & Inventory Optimization | "
    "Powered by Python, Streamlit, FastAPI and PostgreSQL"
)
