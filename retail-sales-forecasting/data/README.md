# Synthetic M5-Like Retail Forecasting Dataset — V2

A synthetic retail dataset designed for a complete **Forecasting + Machine Learning**
portfolio project. The structure is inspired by the M5 Walmart forecasting format,
but the data is synthetic and is NOT Walmart's real data.

## Dataset

- 8 stores
- 28 products
- 224 item-store time series
- Daily data from 2023-01-01 to 2025-12-31
- 28-day validation window
- 28-day final test window
- Events/holidays
- SNAP-like variables
- Weekly selling prices
- Temperature
- Fuel price
- CPI
- Unemployment

## Train / Validation / Test

The split is strictly chronological.

### TRAIN
2023-01-01 → 2025-11-05

Use this data to fit models.

### VALIDATION
2025-11-06 → 2025-12-03

Use this for model comparison and hyperparameter tuning.

### TEST
2025-12-04 → 2025-12-31

The `test.csv` file intentionally contains NO `sales` column.
The actual values are stored separately in `test_actual.csv` for final evaluation.

This simulates a real forecasting task where future sales are unknown.

## Files

- `train_2023.csv`, `train_2024.csv`, `train_2025.csv` — training observations with target `sales`
- `validation.csv` — validation observations with target `sales`
- `test.csv` — final unseen test features; NO target
- `test_actual.csv` — actual test sales for final evaluation
- `calendar.csv` — calendar/events/external variables
- `sell_prices.csv` — weekly selling prices
- `sample_submission.csv` — prediction template
- `README.md` — documentation

## Suggested ML workflow

### 1. Baselines
- Naive forecast
- Seasonal naive (same weekday / previous week)
- Moving average

### 2. Feature engineering

Create target-history features ONLY from information available before
the prediction date:

- lag_1
- lag_7
- lag_14
- lag_28
- rolling_mean_7
- rolling_mean_14
- rolling_mean_28
- rolling_std_28

Also use:
- weekday
- week_of_year
- month
- quarter
- year
- is_weekend
- is_event
- event_name_1
- store_id
- item_id
- dept_id
- cat_id
- sell_price
- temperature
- fuel_price
- cpi
- unemployment
- snap

### 3. Models

Recommended sequence:

1. Seasonal Naive
2. Linear/Ridge Regression
3. Random Forest
4. XGBoost
5. LightGBM

### 4. Evaluation

Use:
- MAE
- RMSE
- WMAPE

Do NOT use a random train/test split.

### 5. Final forecasting

After selecting the model using validation:
- retrain using the appropriate historical data
- generate predictions for `test.csv`
- compare predictions with `test_actual.csv`
- produce a 28-day forecast
- export predictions for Power BI

## Important leakage rule

Do not calculate lag/rolling features using future observations.

For example, a prediction for Monday must not use Monday's actual sales
when calculating a lag or rolling statistic.

## Portfolio project title

**Retail Sales Forecasting Using Machine Learning**

Suggested GitHub structure:

    retail-sales-forecasting/
    ├── data/
    │   ├── train.csv
    │   ├── validation.csv
    │   ├── test.csv
    │   ├── test_actual.csv
    │   ├── calendar.csv
    │   └── sell_prices.csv
    ├── notebooks/
    │   ├── 01_eda.ipynb
    │   ├── 02_feature_engineering.ipynb
    │   ├── 03_model_training.ipynb
    │   └── 04_forecasting_evaluation.ipynb
    ├── src/
    ├── dashboards/
    ├── requirements.txt
    └── README.md


## GitHub upload note

The training data is split by year because the individual files are kept below
25 MB:

- `train_2023.csv`
- `train_2024.csv`
- `train_2025.csv`

You can concatenate these three files in Python when training the model.
