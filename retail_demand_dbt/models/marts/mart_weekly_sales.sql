{{ config(materialized='table') }}

-- Weeks start on the day the warehouse uses for date_trunc('week'):
-- Monday on PostgreSQL, Sunday on BigQuery.
select
    cast({{ dbt.date_trunc('week', 'date') }} as date) as week_start,
    store_id,
    state_id,
    item_id,
    dept_id,
    cat_id,
    sum(total_units_sold) as total_units_sold,
    sum(total_revenue) as total_revenue,
    -- revenue / units, so heavy-selling days count more than slow days
    sum(total_revenue) / nullif(sum(case when average_selling_price is not null then total_units_sold end), 0)
        as average_selling_price,
    sum(sales_records) as sales_records,
    sum(event_days) as event_days
from {{ ref('mart_daily_sales') }}
group by
    cast({{ dbt.date_trunc('week', 'date') }} as date),
    store_id,
    state_id,
    item_id,
    dept_id,
    cat_id