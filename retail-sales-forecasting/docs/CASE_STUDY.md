# Retail Sales Forecasting & Demand Planning — Case Study

## 1. Executive summary

This project develops an end-to-end retail analytics workflow that starts with daily item-store sales forecasting and extends the forecast into practical demand-planning decisions.

The workflow covers:

**Forecast → Validate → Explain error → Segment demand → Quantify uncertainty → Simulate inventory policy → Compare scenarios**

The dataset is synthetic and structurally inspired by M5/Walmart-style retail forecasting. It contains 224 item-store series across 8 stores and 3 years of daily observations.

## 2. Business problem

A forecast is useful only when it supports a decision.

The project addresses two connected questions:

1. How accurately can daily item-store demand be forecast up to 28 days ahead?
2. How can forecast error and demand behavior be translated into replenishment and inventory-planning signals?

## 3. Forecasting approach

The modeling workflow includes:

- Chronological train/validation/test splitting
- Seasonal-naive and moving-average baselines
- Lag and rolling-demand features
- Calendar, store, category, price and external variables
- Random Forest, XGBoost and LightGBM experiments
- MAE, RMSE and WMAPE evaluation
- Walk-forward validation
- Direct vs recursive multi-step forecasting
- SHAP-based model explanation
- Volume-aware forecast fallback policies

## 4. Key forecasting finding

The simple Moving Average baseline achieved the strongest result on the original 28-day validation window:

| Model | MAE | RMSE | WMAPE |
|---|---:|---:|---:|
| Seasonal Naive | 1.679 | 2.706 | 27.51% |
| Moving Average | **1.339** | **2.108** | **21.94%** |
| XGBoost | 1.604 | 2.907 | 23.92% |
| LightGBM | 1.567 | 2.772 | 23.36% |

This is deliberately retained as a portfolio finding: more complex machine-learning models did not automatically outperform a strong simple baseline.

## 5. Forecast robustness

The robustness check evaluates all four main approaches across three chronological 28-day windows. The baseline methods use only the historical training window. XGBoost and LightGBM are fit separately for each window and forecast recursively, so each later forecast day uses prior predictions rather than future actual sales as lag inputs.

| Model | Mean MAE | Mean RMSE | Mean WMAPE |
|---|---:|---:|---:|
| Seasonal Naive | 1.717 | 2.652 | 27.07% |
| Moving Average | **1.527** | **2.472** | **23.45%** |
| XGBoost | 3.629 | 5.243 | 55.87% |
| LightGBM | 3.555 | 5.142 | 54.73% |

The walk-forward results support the same broad observation as the primary holdout: on this synthetic dataset and evaluation setup, the Moving Average baseline remained below the tested ML approaches on mean WMAPE. The result is dataset- and setup-specific rather than a general rule about forecasting model complexity.

The detailed window-level output is stored in `reports/walk_forward_windows.csv`, while `reports/walk_forward_summary.csv` contains the mean and standard deviation for each model.

## 6. Explainability and error analysis

SHAP analysis showed that lagged demand variables were among the strongest contributors in the sampled XGBoost model, particularly:

- lag_7
- lag_14
- lag_1
- lag_28

The project also evaluates errors by forecast horizon, store, category, product and demand-volume segment.

## 7. Demand segmentation

The advanced planning layer profiles each item-store series using historical demand only.

Metrics include:

- Mean daily demand
- Demand standard deviation
- Coefficient of variation
- Zero-demand rate
- Recent 28-day demand
- Early 28-day demand
- Trend ratio

The resulting segments combine volume with demand behavior, for example:

- Low volume / Stable
- Medium volume / Stable
- High volume / Growing
- High volume / Stable
- Medium volume / Growing
- Low volume / Growing

This allows planning policies to be considered in the context of the underlying demand pattern.

## 8. Forecast uncertainty proxy

The project does not claim to produce calibrated probabilistic forecasts.

Instead, it uses historical forecast errors from a 7-day seasonal-naive proxy to summarize:

- MAE
- Error standard deviation
- P90 absolute error
- P95 absolute error

These statistics are then used as inputs to an inventory-policy simulation.

## 9. Inventory planning simulation

The planning layer uses explicit assumptions:

- Lead time: 7 days
- Service level: 95%
- z-value: 1.645
- Baseline planning demand: recent 28-day mean
- Safety-stock input: error-standard-deviation proxy
- Demand scenarios: 90%, 100% and 110% of baseline
- Simulated inventory position: 7 days of baseline demand

Reorder point is calculated as:

**Expected lead-time demand + safety stock**

Because the synthetic dataset has no observed inventory, purchase-order or historical stockout fields, inventory position is simulated. Stockout-risk flags therefore represent the behavior of the policy under the stated assumptions rather than historical stockouts.

## 10. Scenario results

Current planning run:

| Scenario | Simulated inventory | Reorder requirement | Stockout-risk flags |
|---|---:|---:|---:|
| Low demand | 9,753.75 | 11,097.60 | 224 |
| Base demand | 9,753.75 | 12,072.97 | 224 |
| High demand | 9,753.75 | 13,048.35 | 224 |

The high-demand scenario requires the largest replenishment quantity, while the low-demand scenario requires the least.

The purpose of this analysis is to demonstrate the mechanics of translating demand uncertainty into inventory decisions.

## 11. Dashboard

The Streamlit dashboard integrates:

- Executive forecast view
- Forecast diagnostics
- Direct vs recursive model comparison
- Demand segmentation
- Uncertainty profile
- Reorder-point scenarios
- Business exposure

Run:

`streamlit run dashboard/app.py`

## 12. Reproducibility

From the `retail-sales-forecasting` directory:

`python run_all.py`

The pipeline runs:

1. Flagship EDA and baseline analysis
2. Walk-forward validation
3. Advanced demand planning

Planning-only execution:

`python -m advanced_demand_planning.src.run_planning`

## 13. Limitations

The project intentionally documents its limitations:

- The dataset is synthetic.
- Inventory position is simulated.
- No historical purchase orders are available.
- No observed stockout events are available.
- The uncertainty layer is an error proxy, not calibrated probabilistic forecasting.
- The demand-segmentation and fallback policies are demonstration policies, not universal retail rules.
- Test-period performance is specific to this synthetic dataset.

## 14. Portfolio takeaway

The main lesson is not that one model is universally superior.

The project demonstrates a complete analytical workflow in which:

**model performance → forecast diagnostics → demand behavior → uncertainty → operational decision**

That progression is the main portfolio value of the project.
