-- Fails if any row of raw sales has negative units.
select *
from {{ ref('stg_daily_sales') }}
where sales < 0