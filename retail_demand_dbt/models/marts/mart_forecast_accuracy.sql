{{ config(materialized='table') }}

-- How good were the forecasts on days that have already happened?
-- WAPE = sum(|actual - predicted|) / sum(actual); lower is better.
select
    model_name,
    model_version,
    run_id,
    store_id,
    dept_id,
    count(*) as forecast_rows,
    sum(actual_units) as actual_units,
    sum(predicted_units) as predicted_units,
    sum(abs(actual_units - predicted_units)) as absolute_error,
    sum(abs(actual_units - predicted_units)) / nullif(sum(actual_units), 0) as wape
from {{ ref('stg_demand_forecasts') }}
where is_backtest and actual_units is not null
group by
    model_name,
    model_version,
    run_id,
    store_id,
    dept_id