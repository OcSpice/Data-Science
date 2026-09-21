# Retail Sales Forecasting & Demand Planning

An end-to-end time-series forecasting and machine-learning portfolio project built around a synthetic M5-style retail dataset.

> The dataset is synthetic and structurally inspired by the M5/Walmart forecasting format. It is not Walmart proprietary data.

## Business problem

Retailers need reliable short-term demand forecasts for inventory planning, replenishment, staffing and promotion decisions.

**Question:** How accurately can daily item-store sales be forecast 28 days ahead, and where do forecast errors concentrate?

## Workflow

1. Data validation and chronological splitting
2. Exploratory data analysis
3. Time-series decomposition and seasonality analysis
4. Baselines: Seasonal Naive and Moving Average
5. Time-aware feature engineering: lag 1/7/14/28, rolling statistics, calendar/cyclical variables, store/product/category, price and external variables
6. Machine-learning models: Random Forest, XGBoost, LightGBM
7. Evaluation: MAE, RMSE, WMAPE
8. Error analysis by store, category, product and forecast horizon
9. Final 28-day forecast
10. Business recommendations
11. Dashboard-ready outputs

## Data split

| Split | Dates | Purpose |
|---|---|---|
| Train | 2023-01-01 → 2025-11-05 | Model fitting |
| Validation | 2025-11-06 → 2025-12-03 | Model selection/tuning |
| Test | 2025-12-04 → 2025-12-31 | Final unseen evaluation |

The test feature file contains no sales target; actual test sales are stored separately for final evaluation.

## Repository structure

```
retail-sales-forecasting/
├── data/
├── src/
│   ├── data.py
│   ├── features.py
│   ├── baselines.py
│   ├── metrics.py
│   └── modeling.py
├── dashboard/
├── tests/
├── reports/
└── requirements.txt
```

## Run locally

```bash
pip install -r requirements.txt
pytest -q
```

The synthetic dataset used by this project is intentionally kept outside the GitHub source tree when file-size limits make direct upload impractical. Place the dataset files in `data/` using the supplied dataset ZIP.

## Leakage control

No random train/test split is used. Lag and rolling features are calculated only from observations available before the prediction date.

## Portfolio extensions

- Walk-forward validation
- Hyperparameter tuning with time-series cross-validation
- SHAP model explainability
- Store/category/product error heatmaps
- Forecast uncertainty and prediction intervals
- Power BI/Tableau executive dashboard
- Inventory safety-stock simulation
