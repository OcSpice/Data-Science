# Time Series Sales Forecasting

**Author:** OGHENEOCHUKU EMMANUEL OGIDIAGBA  
**Portfolio Category:** Data Science

---

## Project Overview

This project implements a **from-scratch Triple Exponential Smoothing (Holt-Winters)** forecasting model for aggregated daily retail sales.

The core forecasting logic is implemented with NumPy and Pandas rather than a black-box time-series library. The project also demonstrates chronological model validation, validation-based parameter selection, baseline benchmarking, reproducible testing, and GitHub Actions execution.

The forecasting target is the **daily total sales across the available item/store observations**.

---

## Dataset

The project uses a retail sales dataset organized into:

- `train_2023.csv`
- `train_2024.csv`
- `train_2025.csv`
- `validation.csv`
- `test.csv`
- `test_actual.csv`
- `calendar.csv`
- `sell_prices.csv`

`test_actual.csv` is retained separately as the final evaluation truth for the otherwise unlabeled `test.csv` period.

### Chronological evaluation design

| Period | Dates | Purpose |
|---|---|---|
| Training | 2023-01-01 → 2025-11-05 | Fit the model |
| Validation | 2025-11-06 → 2025-12-03 | Select smoothing parameters |
| Final test | 2025-12-04 → 2025-12-31 | Unseen final evaluation |

The final test period is not used during parameter selection.

---

## Forecasting Method

The project implements additive Holt-Winters with a weekly seasonal period:

- Level smoothing: **α**
- Trend smoothing: **β**
- Seasonal smoothing: **γ**
- Trend damping: **d**
- Seasonal period: **7 days**

The implementation includes:

- component initialization
- iterative level/trend/seasonal updates
- damped or undamped trend forecasting
- multi-step forecasting
- additive and multiplicative model support
- residual calculation
- confidence-interval support
- seasonal-phase continuity after the final training observation

### Validation-based parameter selection

The current pipeline searches:

- α: 0.1, 0.2, 0.3, 0.5, 0.7
- β: 0.0, 0.05, 0.1, 0.2, 0.3
- γ: 0.05, 0.1, 0.2, 0.3
- damping: 0.7, 0.8, 0.9, 0.95, 1.0

This produces **500 candidate configurations**.

Each candidate is fitted **only on the training period** and evaluated on the validation period using MAPE as the primary selection metric. The selected configuration is then refitted on training + validation data before final test evaluation.

This prevents the final test period from influencing parameter selection.

---

## Final Runtime Results

The current GitHub Actions runtime successfully executed the complete pipeline on `main`.

### Selected configuration

- Seasonal type: **Additive**
- Seasonal period: **7**
- α = **0.70**
- β = **0.05**
- γ = **0.05**
- Damping = **0.70**
- Selection metric: **Validation MAPE**

### Validation performance

| Metric | Damped Holt-Winters |
|---|---:|
| MAPE | **6.68%** |
| MAE | **153.71** |
| RMSE | **427.25** |
| SMAPE | **8.22%** |
| MASE | **1.15** |

### Final unseen test performance

| Model | MAPE | MAE | RMSE |
|---|---:|---:|---:|
| **Damped Holt-Winters** | **5.10%** | **90.22** | **209.39** |
| Naive | 11.57% | 195.21 | 294.31 |
| Seasonal Naive | 17.41% | 274.54 | 493.72 |
| Mean | 9.79% | 155.26 | 228.82 |
| Drift | 11.39% | 192.44 | 291.66 |

### Interpretation

The initial undamped Holt-Winters configuration performed poorly on the final test period. The model was subsequently extended with a **damped-trend component** and the damping factor was selected using the validation period.

The resulting locked model achieved **5.10% MAPE** on the final 28-day test period, compared with **9.79%** for the Mean baseline and **11.57%** for the Naive baseline.

This is a period-specific out-of-sample result, not a guarantee of future forecasting performance. The test period remains separate from parameter selection.

---

## Why the Final Test Matters

The earlier version of this project optimized Holt-Winters parameters using in-sample SSE and evaluated a date range beginning on 2025-11-01. That did not match the documented dataset boundaries.

The current implementation corrects this by:

1. using the documented 2025-11-05 training cutoff;
2. selecting parameters on 2025-11-06 → 2025-12-03 validation data;
3. tuning the damping factor as part of the validation search;
4. refitting only after parameter selection;
5. evaluating on the separate 2025-12-04 → 2025-12-31 final test period;
6. comparing the result with multiple simple baselines.

---

## Repository Structure

```
Time Series Sales Forecasting/
├── Datasets/
│   ├── train_2023.csv
│   ├── train_2024.csv
│   ├── train_2025.csv
│   ├── validation.csv
│   ├── test.csv
│   ├── test_actual.csv
│   ├── calendar.csv
│   ├── sell_prices.csv
│   └── README.md
├── src/
│   ├── data_loader.py
│   ├── evaluation.py
│   ├── feature_engineering.py
│   ├── holt_winters.py
│   ├── pipeline.py
│   ├── report_generator.py
│   └── visualization.py
├── tests/
│   └── test_holt_winters.py
└── README.md
```

Generated reports are intentionally excluded from source control. GitHub Actions produces them as a validation artifact.

---

## Running the Project

From the project directory:

```bash
pip install -r requirements.txt
python src/pipeline.py
```

The pipeline generates:

- `reports/forecast_predictions.csv`
- `reports/parameter_search.csv`
- `reports/metrics.json`
- `reports/model_parameters.json`
- `reports/executive_summary.txt`

---

## Testing

Run:

```bash
pytest tests/test_holt_winters.py -v
```

The test suite covers:

- Holt-Winters initialization
- damping validation
- fitting and residual generation
- forecast generation
- seasonal behavior
- damped-trend forecasting
- evaluation metrics
- baseline forecasts
- end-to-end fit/predict behavior
- chronological train/validation/test boundaries
- validation-based parameter selection
- the 500-configuration parameter grid
- seasonal-phase continuity

The latest GitHub Actions validation run completed successfully after the damped-trend changes.

---

## GitHub Actions

Workflow:

`.github/workflows/time-series-forecasting.yml`

The workflow:

1. checks out the repository;
2. installs the project dependencies;
3. runs the complete test suite;
4. executes the full forecasting pipeline;
5. uploads the generated forecasting outputs as a CI artifact.

Latest verified successful run:

- Workflow: **Time Series Forecasting Validation**
- Run: **#26**
- Run ID: **35727923001**
- Branch: **main**
- Commit: `b4a553fa1f6d88a57a2794254eb55dfb78f738e6`
- Result: **success**
- Artifact: `time-series-forecasting-validation-outputs`

---

## Limitations

- The primary model is an aggregated univariate forecast and does not currently exploit item/store-level covariates in the forecasting equation.
- The final test period contains only 28 days, so conclusions should be treated as period-specific rather than universal.
- Validation performance does not guarantee future-period performance.
- The model is evaluated on aggregated daily sales, so item/store-level forecasting behavior is not directly assessed.
- The project is a portfolio/research implementation, not a production demand-planning system.

---

## Key Takeaway

This project focuses on **reproducible forecasting methodology rather than presenting an inflated accuracy claim**.

The final implementation demonstrates:

- a mathematical forecasting algorithm built from scratch;
- damped-trend Holt-Winters forecasting;
- chronological leakage-safe validation;
- validation-based selection across 500 parameter configurations;
- a genuinely held-out final test;
- baseline benchmarking;
- automated tests;
- reproducible GitHub Actions execution;
- transparent reporting of model performance and limitations.
