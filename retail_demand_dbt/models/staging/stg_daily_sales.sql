{{ config(materialized='view') }}

select
    id,
    series_id,
    item_id,
    dept_id,
    cat_id,
    store_id,
    state_id,
    d,
    sales,
    date,
    wm_yr_wk,
    sell_price,
    weekday,
    wday,
    month,
    year,
    event_name_1,
    event_type_1,
    event_name_2,
    event_type_2,
    "snap_CA" as snap_ca,
    "snap_TX" as snap_tx,
    "snap_WI" as snap_wi

from {{ source('retail', 'daily_sales') }}
