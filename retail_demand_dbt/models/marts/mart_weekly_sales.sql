{{ config(materialized='table') }}

select
    date_trunc('week', date)::date as week_start,
    store_id,
    state_id,
    item_id,
    dept_id,
    cat_id,
    sum(total_units_sold) as total_units_sold,
    avg(average_selling_price) as average_selling_price,
    sum(sales_records) as sales_records,
    sum(event_days) as event_days

from {{ ref('mart_daily_sales') }}

group by
    date_trunc('week', date)::date,
    store_id,
    state_id,
    item_id,
    dept_id,
    cat_id
