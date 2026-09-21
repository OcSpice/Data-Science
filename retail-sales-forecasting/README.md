# Retail Sales Forecasting & Demand Planning

End-to-end time-series forecasting and machine-learning portfolio project using a synthetic M5-style retail dataset.

## Business problem

Retailers need reliable short-term demand forecasts for inventory planning, replenishment, staffing and promotion decisions.

**Question:** How accurately can daily item-store sales be forecast 28 days ahead, and where do forecast errors concentrate?

## Workflow

Data validation → EDA → seasonality/decomposition → baselines → time-aware feature engineering → Random Forest/XGBoost/LightGBM → MAE/RMSE/WMAPE → walk-forward validation → error analysis → 28-day forecast → business recommendations → dashboard.

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

This is an important analytical finding: model complexity is not automatically an improvement. The project now includes walk-forward evaluation utilities so model selection can be checked across multiple chronological windows rather than one validation period.

## Walk-forward validation

`src/walk_forward.py` provides:

- Multiple chronological forecast windows
- 28-day horizons by default
- Seasonal-naive evaluation without future-data leakage
- Window-level MAE, RMSE and WMAPE
- Mean and standard deviation across windows

Run:

`python -m src.run_walk_forward`

The runner writes:

- `reports/walk_forward_windows.csv`
- `reports/walk_forward_summary.csv`

These outputs are intended to support a more defensible model-selection decision before the final test forecast.

## Feature engineering

- Lag 1/7/14/28
- Rolling mean 7/14/28
- Rolling standard deviation 28
- Day-of-week and month cyclical features
- Store, state, item, department and category
- Price and economic/weather variables
- Event/SNAP indicators

## EDA utilities

`src/eda.py` provides reusable summaries for:

- Dataset coverage and dimensions
- Daily demand
- Product category demand
- Store demand
- Weekday demand patterns

This keeps the exploratory analysis reproducible instead of relying only on notebook screenshots.

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
- Walk-forward validation
- Reproducible metrics
- CSV reporting artifacts
- Streamlit dashboard scaffold

## Dataset note

The data is synthetic and structurally inspired by M5/Walmart-style retail forecasting. It is not Walmart proprietary data.


## Flagship layer

The project now extends beyond a single train/validation benchmark into a portfolio-style forecasting workflow.

### EDA and visual analysis
`src/flagship_analysis.py` generates reproducible views for daily sales trend, weekly and monthly seasonality, store and category comparisons, price vs sales, and event effects.

### Walk-forward validation
Three historical 28-day windows are evaluated chronologically. The current Seasonal Naive baseline produced:

| Window | MAE | RMSE | WMAPE |
|---|---:|---:|---:|
| 2025-09-01 → 2025-09-28 | 1.742 | 2.606 | 26.11% |
| 2025-10-06 → 2025-11-02 | 1.730 | 2.643 | 27.59% |
| 2025-11-06 → 2025-12-03 | 1.679 | 2.706 | 27.51% |

The variation across windows is evidence that model quality should not be judged from one holdout period.

### Model improvement
A log-target XGBoost configuration was tested against the original validation window. Its validation WMAPE was 25.08%, compared with 23.92% for the earlier XGBoost configuration and 21.94% for the Moving Average baseline. This does not replace the baseline; it demonstrates that target transformation is an experiment, not an assumed improvement.

The final 28-day recursive log-XGBoost test forecast produced MAE 3.268, RMSE 4.719 and WMAPE 49.04% on the synthetic held-out test period.

### SHAP explainability
`src/shap_analysis.py` produces a SHAP beeswarm plot and ranked mean absolute SHAP values. In the sampled data, lag features—especially lag 7, lag 14, lag 1 and lag 28—were the strongest contributors, followed by category/store and calendar/economic variables.

### Forecast diagnostics
The flagship workflow includes aggregate actual-vs-predicted 28-day analysis, horizon-level WMAPE, recursive multi-step forecasting, and a direct multi-step forecasting implementation for controlled comparison.

### Dashboard
The Streamlit dashboard is designed around the hiring-manager view: Sales → Forecast → Error → Store → Category → Business implication.

Run locally with: `streamlit run dashboard/app.py`

The project intentionally keeps the Moving Average baseline visible. Model complexity is treated as an experiment that must be justified by measured improvement.
