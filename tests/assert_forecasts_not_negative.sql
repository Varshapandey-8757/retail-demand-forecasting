-- Fails if a model predicted negative demand.
select *
from {{ ref('stg_demand_forecasts') }}
where predicted_units < 0