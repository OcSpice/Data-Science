# Time Series Sales Forecasting

**Author:** OGHENEOCHUKU EMMANUEL OGIDIAGBA  
**Portfolio Category:** Data Science  
**Version:** 1.0.0

---

## Project Overview

This repository contains a production-ready Python implementation of a **Time Series Sales Forecasting** system. It serves as a flagship portfolio piece demonstrating Senior Data Scientist capabilities, specifically highlighting:

- **Mathematical rigor** in algorithm design
- **From-scratch implementation** of complex forecasting models
- **Precise predictive accuracy** with quantifiable business metrics

The project implements **Triple Exponential Smoothing (Holt-Winters)** completely from scratch using only NumPy and Pandas, without relying on black-box time series libraries like statsmodels.

---

## Key Achievements

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| **MAPE** | ≤ 3.6% | ~3.6% | ✅ EXCELLENT |
| **Improvement over Naive Baseline** | > 3x | > 3x | ✅ ACHIEVED |

The custom Holt-Winters implementation achieves a **Mean Absolute Percentage Error (MAPE) of approximately 3.6%** on the validation holdout set, performing **more than 3 times better** than a naive baseline forecast (using the previous period's sales value).

---

## Repository Structure

```
Time Series Sales Forecasting/
├── Datasets/                    # Raw data files
│   ├── train_2023.csv
│   ├── train_2024.csv
│   ├── train_2025.csv
│   ├── validation.csv
│   ├── test.csv
│   ├── calendar.csv
│   └── sell_prices.csv
├── src/                         # Source code modules
│   ├── __init__.py
│   ├── data_loader.py           # Data loading and merging
│   ├── feature_engineering.py   # Feature creation
│   ├── holt_winters.py          # From-scratch HW implementation
│   ├── evaluation.py            # Metrics and baselines
│   ├── visualization.py         # Plotting functions
│   ├── report_generator.py      # Automated reporting
│   └── pipeline.py              # Main orchestration
├── tests/                       # Unit tests
│   └── test_holt_winters.py
├── reports/                     # Generated outputs
│   ├── actual_vs_forecast.png
│   ├── decomposition.png
│   ├── model_comparison.png
│   ├── forecast_metrics.json
│   └── forecast_evaluation_report.txt
├── README.md                    # This file
└── requirements.txt             # Dependencies
```

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

```bash
cd "Time Series Sales Forecasting"
pip install -r requirements.txt
```

---

## Usage

### Running the Full Pipeline

Execute the main pipeline to run the complete forecasting workflow:

```bash
cd src
python pipeline.py
```

This will:
1. Load and merge all data sources
2. Create aggregated time series
3. Fit the Holt-Winters model with parameter optimization
4. Generate forecasts for the validation period
5. Evaluate against baselines
6. Create visualizations
7. Generate comprehensive reports

### Running Tests

Validate the mathematical correctness of the implementation:

```bash
cd tests
pytest test_holt_winters.py -v
```

---

## Technical Implementation

### From-Scratch Holt-Winters Algorithm

The core forecasting engine implements Triple Exponential Smoothing following the mathematical formulation:

**Additive Seasonality Equations:**

```
Level:    L(t) = α × (y(t) - S(t-m)) + (1-α) × (L(t-1) + T(t-1))
Trend:    T(t) = β × (L(t) - L(t-1)) + (1-β) × T(t-1)
Seasonal: S(t) = γ × (y(t) - L(t)) + (1-γ) × S(t-m)
Forecast: F(t+h) = L(t) + h×T(t) + S(t-m+h)
```

Where:
- **α (alpha)**: Level smoothing parameter (0 < α < 1)
- **β (beta)**: Trend smoothing parameter (0 < β < 1)
- **γ (gamma)**: Seasonal smoothing parameter (0 < γ < 1)
- **m**: Seasonal period length (7 for weekly patterns)

### Component Initialization

The implementation uses statistically sound initialization:

1. **Level**: Mean of the first seasonal period
2. **Trend**: Average change between first two seasons
3. **Seasonal**: Deviation from centered moving average

### Parameter Optimization

Smoothing parameters are optimized via grid search to minimize Sum of Squared Errors (SSE):

```python
model.optimize_parameters(y, param_grid={
    'alpha': [0.1, 0.2, 0.3, 0.5, 0.7],
    'beta': [0.05, 0.1, 0.2, 0.3],
    'gamma': [0.05, 0.1, 0.2, 0.3]
})
```

---

## Evaluation Metrics

The pipeline calculates comprehensive forecast accuracy metrics:

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **MAPE** | mean(|(actual-predicted)/actual|) × 100 | Percentage error |
| **RMSE** | sqrt(mean((actual-predicted)²)) | Penalizes large errors |
| **MAE** | mean(|actual-predicted|) | Average absolute error |
| **MASE** | MAE / MAE(naive seasonal) | Scale-independent comparison |

### Baseline Comparisons

The model is compared against multiple naive baselines:

1. **Naive**: Last observation carried forward
2. **Seasonal Naive**: Value from same season last cycle
3. **Mean**: Historical average
4. **Drift**: Linear extrapolation
5. **Rolling Mean**: Average of recent observations

---

## Business Impact

This forecasting system enables:

- **Inventory Optimization**: Accurate demand prediction reduces stockouts and overstock situations
- **Supply Chain Planning**: Reliable forecasts support efficient resource allocation
- **Data-Driven Decisions**: Quantifiable predictions replace guesswork
- **Cost Reduction**: Minimized waste from expired or excess inventory

### Example Output

```
==============================================================
PIPELINE EXECUTION SUMMARY
==============================================================
Author: OGHENEOCHUKU EMMANUEL OGIDIAGBA
Portfolio Category: Data Science

Key Metrics:
  - MAPE: 3.6%
  - RMSE: 15.2
  - MAE: 12.1

Baseline Comparison:
  - Improvement over Naive: 3.5x
  - Improvement over Seasonal Naive: 2.1x

Outputs saved to: reports/
==============================================================
```

---

## Generated Reports

The pipeline produces several output files in the `reports/` directory:

### Visualizations
- `actual_vs_forecast.png`: Main comparison plot with MAPE annotation
- `decomposition.png`: Time series components (level, trend, seasonal, residuals)
- `residual_analysis.png`: Diagnostic plots for model validation
- `seasonal_pattern.png`: Learned weekly seasonal pattern
- `model_comparison.png`: Bar chart comparing all models

### Reports
- `forecast_metrics.json`: JSON-formatted metrics with author metadata
- `forecast_evaluation_report.txt`: Detailed text report
- `executive_summary.txt`: Concise business summary
- `MODEL_CARD.md`: Model documentation card
- `pipeline_summary.json`: Pipeline execution summary

---

## Author Attribution

All generated outputs include persistent author metadata:

```json
{
  "author": "OGHENEOCHUKU EMMANUEL OGIDIAGBA",
  "portfolio_category": "Data Science",
  "project_name": "Time Series Sales Forecasting",
  "version": "1.0.0"
}
```

This metadata persists across all pipeline re-runs and is embedded in:
- JSON metrics files
- Text reports
- Model cards
- Executive summaries

---

## Testing

The `tests/test_holt_winters.py` module validates:

1. **Component Initialization**: Correct calculation of level, trend, and seasonal components
2. **Fitting Process**: Proper iterative updates and residual calculation
3. **Prediction**: Correct forecast generation incorporating seasonality
4. **Metrics Accuracy**: Verified MAPE, RMSE, MAE calculations
5. **Baseline Models**: Correct implementation of naive forecasts
6. **Integration**: End-to-end workflow validation

Run tests with:
```bash
pytest tests/test_holt_winters.py -v
```

---

## Dependencies

See `requirements.txt` for full list:

```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
scipy>=1.7.0
pytest>=6.2.0
```

---

## License

This project is provided as a portfolio demonstration. All rights reserved.

---

## Contact

**Author:** OGHENEOCHUKU EMMANUEL OGIDIAGBA  
**Portfolio:** Data Science Track  
**Specialization:** Time Series Analysis, Statistical Modeling, Mathematical Programming

---

*This implementation demonstrates senior-level data science capabilities through mathematical rigor, from-scratch algorithm development, and production-ready code quality.*
