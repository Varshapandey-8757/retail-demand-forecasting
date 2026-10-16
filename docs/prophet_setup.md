# Prophet Setup Notes (Model A)

## Environment
- OS: Windows
- Branch: feature/prophet-forecast
- Core packages:
  - prophet (1.5.0)
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - jupyter

## Project layout for Prophet work
- `data/raw/`: M5 raw CSV files (ignored by git)
- `notebooks/`: Exploration, baseline fitting, evaluation
- `pipelines/`: Training and forecast export scripts
- `tests/`: Automated checks on forecast format and dates

## Planned series count
- Prophet fits one model per time series.
- We will select 10 to 20 representative product/store series rather than all ~30,000.