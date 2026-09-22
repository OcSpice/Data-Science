# Customer Churn Prediction & Retention Economics

A portfolio-grade data science project combining **customer churn classification, model comparison, explainability, probability-threshold analysis, and retention economics**.

## Business question

> Which customers are most likely to churn, how reliable are the predictions, and how does the intervention threshold change the size and economics of a retention campaign?

The project separates:
- observed churn outcomes from model-predicted risk;
- revenue exposure from realized revenue loss;
- SHAP model explanations from causal claims; and
- scenario assumptions from observed business results.

## Workflow

```
Customer data
    ↓
Validation + privacy handling
    ↓
Train/test split
    ↓
Train-only preprocessing
    ↓
Majority baseline + Logistic Regression + Random Forest
    ↓
ROC-AUC / PR-AUC / Precision / Recall / F1
    ↓
Probability threshold analysis
    ↓
SHAP TreeExplainer
    ↓
Risk segmentation
    ↓
Revenue exposure
    ↓
Retention cost/effectiveness scenarios
```

## Modeling methodology

### Leakage-safe preprocessing

The train/test split occurs **before** model preprocessing is fitted.

Categorical variables use:

`OneHotEncoder(handle_unknown="ignore")`

Numeric variables use median imputation. Logistic Regression additionally uses `StandardScaler`.

### Models

1. **Majority-class baseline** — establishes the performance floor.
2. **Logistic Regression** — interpretable linear baseline.
3. **Random Forest** — nonlinear class-balanced model tuned with ROC-AUC cross-validation; it is used for downstream risk segmentation and SHAP explainability because its recall is slightly higher than Logistic Regression on the held-out test set.

### Evaluation

The project reports:
- Accuracy
- Precision
- Recall
- F1
- ROC-AUC
- PR-AUC
- Confusion matrix

Accuracy is not treated as the sole success criterion for a retention use case.

### Verified held-out test results

The GitHub Actions runtime generated the following results:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Majority baseline | 60.70% | 0.00% | 0.00% | 0.00% | 0.500 | 0.393 |
| Logistic Regression | **72.33%** | **61.39%** | 79.72% | **69.37%** | **0.800** | **0.694** |
| Random Forest | 71.46% | 60.23% | **80.59%** | 68.94% | 0.794 | 0.685 |

The Logistic Regression model has the stronger ROC-AUC and F1 on this split, while Random Forest has slightly higher recall. The project therefore does not claim that Random Forest is universally superior; it is retained for the downstream risk/economic analysis and model-specific SHAP explanation.

At the default **0.70 risk threshold**, the Random Forest identifies **1,520 customers (21.71%)** with **$130,890.08 in monthly revenue exposure**, equivalent to **$1.57M annualized exposure** if the monthly amount were sustained. This is a prioritization metric, not a forecast of realized revenue loss.

## Threshold analysis

The pipeline evaluates churn-probability thresholds:

`0.30, 0.40, 0.50, 0.60, 0.70, 0.80`

For each threshold it reports:
- customers targeted;
- target share;
- monthly revenue exposure; and
- annualized revenue exposure.

This makes threshold selection an explicit business-policy trade-off instead of an arbitrary fixed cutoff.

## Retention economics

The scenario model uses explicit assumptions:

- intervention cost: **$20 per targeted customer**;
- intervention effectiveness: **10%, 20%, 30%** scenarios.

For each threshold/scenario combination:

`Expected retained monthly value = revenue exposure × assumed effectiveness`

`Expected net monthly value = expected retained monthly value − intervention cost`

These are **hypothetical scenarios**, not measured intervention outcomes.

Revenue exposure is the sum of `MonthlyCharges` for customers above a selected probability threshold. It is **not** a claim that all exposed revenue will be lost.

## Explainability

The Random Forest is explained with genuine:

`shap.TreeExplainer`

SHAP is computed on held-out test data. One-hot encoded contributions are aggregated back to source variables for global interpretation.

A local example is also generated to demonstrate how individual encoded features contributed to a single prediction.

SHAP explains model behavior; it does not establish that a feature causes churn.

## Outputs

The pipeline generates the following files in the local `outputs/` directory:

| Output | Purpose |
|---|---|
| `model_metrics.json` | Baseline/model comparison plus threshold and scenario tables |
| `business_impact.json` | Revenue exposure and retention economics |
| `shap_summary.json` | Global and local SHAP explanations |
| `full_pipeline_report.json` | Consolidated analytical report |
| `executive_summary.txt` | Executive-readable summary |
| `churn_model.joblib` | Serialized model artifacts |

These generated artifacts are intentionally excluded from source control to prevent stale model files and environment-specific binaries from becoming the repository source of truth. GitHub Actions generates and uploads the current six-file validation artifact on successful runs.

## Run locally

```bash
cd "Customer Churn Prediction Model"
python -m pip install -r requirements.txt
pytest -q tests/test_pipeline.py
python src/pipeline.py
```

## Interpretation cautions

This is a **portfolio-grade analytical prototype**, not a production deployment. It does not claim to provide:
- model monitoring;
- drift detection;
- a model registry;
- scheduled inference;
- API serving; or
- causal estimates of retention impact.

## Author

**OGHENEOCHUKO EMMANUEL OGIDIAGBA**

Data Science Portfolio — Customer Churn Prediction & Retention Economics
