{{ config(materialized='view') }}

select
    id,
    run_id,
    model_name,
    model_version,
    series_id,
    store_id,
    item_id,
    dept_id,
    cat_id,
    forecast_date,
    predicted_units,
    actual_units,
    is_backtest

from {{ source('retail', 'demand_forecasts') }}