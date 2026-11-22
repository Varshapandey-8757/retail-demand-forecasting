{{ config(materialized='table') }}

select
    date,
    store_id,
    state_id,
    item_id,
    dept_id,
    cat_id,
    sum(sales) as total_units_sold,
    avg(sell_price) as average_selling_price,
    sum(sales * sell_price) as total_revenue,
    count(*) as sales_records,
    sum(case when event_name_1 is not null then 1 else 0 end) as event_days
from {{ ref('stg_daily_sales') }}
group by
    date,
    store_id,
    state_id,
    item_id,
    dept_id,
    cat_id