# Retail Sales Forecasting & Demand Planning

End-to-end time-series forecasting and machine-learning portfolio project using a synthetic M5-style retail dataset.

## Business problem

Retailers need reliable short-term demand forecasts for inventory planning, replenishment, staffing and promotion decisions.

**Question:** How accurately can daily item-store sales be forecast 28 days ahead, and where do forecast errors concentrate?

## Workflow

Data validation → EDA → seasonality/decomposition → baselines → time-aware feature engineering → Random Forest/XGBoost/LightGBM → MAE/RMSE/WMAPE → error analysis → 28-day forecast → business recommendations → dashboard.

## Chronological split

- Train: 2023-01-01 to 2025-11-05
- Validation: 2025-11-06 to 2025-12-03
- Test: 2025-12-04 to 2025-12-31

The test feature file contains no target; actual test sales are stored separately.

## Validation result

The first benchmark is intentionally honest: the simple **Moving Average baseline outperformed the tree models on this synthetic dataset** during the 28-day validation window.

| Model | MAE | RMSE | WMAPE |
|---|---:|---:|---:|
| Seasonal Naive | 1.679 | 2.706 | 27.51% |
| Moving Average | **1.339** | **2.108** | **21.94%** |
| XGBoost | 1.604 | 2.907 | 23.92% |
| LightGBM | 1.567 | 2.772 | 23.36% |

This is an important analytical finding: model complexity is not automatically an improvement. The next iteration should use walk-forward validation and tuned multi-step forecasting before selecting a production model.

## Feature engineering

- Lag 1/7/14/28
- Rolling mean 7/14/28
- Rolling standard deviation 28
- Day-of-week and month cyclical features
- Store, state, item, department and category
- Price and economic/weather variables
- Event/SNAP indicators

## Error analysis

The project includes breakdowns by store, category, product and forecast horizon, plus feature-importance outputs for XGBoost and LightGBM.

## 28-day forecast

A Seasonal Naive 28-day forecast is included as a transparent benchmark for the unseen test period. On the held-out test period it produced:

- MAE: **3.091**
- RMSE: **5.380**
- WMAPE: **48.94%**

The benchmark is intentionally separated from the validation model-selection result.

## Engineering quality

- Modular Python
- Automated tests
- Leakage-aware chronological splitting
- Reproducible metrics
- CSV reporting artifacts
- Streamlit dashboard scaffold

## Dataset note

The data is synthetic and structurally inspired by M5/Walmart-style retail forecasting. It is not Walmart proprietary data.
