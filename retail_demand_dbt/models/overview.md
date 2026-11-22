{% docs __overview__ %}

# Retail Demand Forecasting - data model

Walmart M5 sales (item x store x day) are loaded by the Python ETL into
`daily_sales`. The forecasting pipelines write predictions and backtests into
`demand_forecasts`. dbt turns both into clean staging views and analysis marts:

- **staging** - `stg_daily_sales`, `stg_demand_forecasts` (rename/clean only)
- **marts** - daily / weekly / monthly sales, store-department weekly sales,
  and forecast accuracy by model, store and department

Use the **lineage graph** (blue icon, bottom right) to follow a column from the
raw tables to the marts.

{% enddocs %}
