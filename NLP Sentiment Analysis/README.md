# NLP Sentiment Analysis & Customer Feedback Analytics

**Portfolio Category:** Data Science  
**Dataset:** 12,000 customer reviews  
**Task:** Multiclass sentiment classification (Positive, Neutral, Negative)

## Overview

This project turns unstructured customer reviews into two outputs:

1. **Sentiment prediction** using classical NLP/ML models.
2. **Customer-feedback theme analytics** showing which recurring topics are associated with negative sentiment.

The project is intentionally evaluation-first: model performance is measured on held-out data, while business findings are described as observed associations rather than causal root causes.

## Workflow

```
Customer Reviews
      ↓
Schema & data-quality checks
      ↓
Text cleaning + PII anonymization
      ↓
Stratified train/test split
      ↓
TF-IDF fit on training text only
      ↓
Model comparison
  ├─ Majority baseline
  ├─ Multinomial Naive Bayes
  ├─ Logistic Regression
  └─ Linear SVM
      ↓
Accuracy / Precision / Recall / Macro-F1
ROC-AUC / PR-AUC / Confusion Matrix
      ↓
Customer-feedback theme analysis
      ↓
Category × sentiment reporting
```

## Leakage Control

The original implementation fitted TF-IDF on the complete dataset before the train/test split. That has been removed.

The rebuilt evaluation follows:

```
Raw reviews
    ↓
Stratified 80/20 split
    ↓
Training reviews → TF-IDF fit_transform
Test reviews     → TF-IDF transform
    ↓
Model evaluation on held-out test set
```

Therefore, vocabulary and IDF statistics are learned from training text only.

## Model Comparison

The evaluation includes:

- Majority-class baseline
- Multinomial Naive Bayes
- Logistic Regression
- Linear SVM

Reported metrics:

- Accuracy
- Macro Precision
- Macro Recall
- **Macro-F1**
- Weighted-F1
- Multiclass ROC-AUC
- Macro PR-AUC
- Confusion matrix
- Per-class classification report

Macro-F1 is emphasized because the sentiment classes are not perfectly balanced.

## Data Quality Checks

The pipeline reports:

- Total records
- Sentiment distribution
- Average rating
- Exact duplicate review rows
- Number of unique duplicated review texts
- Duplicate texts with conflicting sentiment labels
- Sentiment distribution by product category

Duplicate analysis is especially important for this dataset because repeated or templated reviews can make random train/test evaluation look artificially strong.

## Customer Feedback Theme Analysis

The previous "root-cause analysis" terminology has been replaced with **theme analysis**.

Themes currently include:

- Customer Support
- Billing
- Product Quality
- Performance
- Integration
- Pricing
- Usability
- Marketing Expectations

For each theme, the pipeline reports:

- Number of reviews mentioning the theme
- Share of all reviews
- Number of negative reviews
- Negative rate within the theme
- Share of all negative reviews
- Top product categories associated with the theme

These are **descriptive associations**. Frequent mention of a theme does not prove that the theme caused dissatisfaction.

## Text Preprocessing

The existing preprocessing layer handles:

- HTML removal
- URL and email cleanup
- Phone-number anonymization
- Special-character normalization
- Tokenization
- Stopword removal
- Lemmatization

## Outputs

The pipeline generates:

- Sentiment distribution visualization
- Negative-review word cloud
- Top negative phrase visualization
- JSON analytical report
- Human-readable text summary

## Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the pipeline:

```bash
cd "NLP Sentiment Analysis"
python src/main_pipeline.py
```

Run tests:

```bash
pytest tests/test_pipeline.py -v
```

## Limitations

- This is a portfolio analytical prototype, not a production NLP service.
- Results depend on the supplied customer-review dataset.
- Random train/test evaluation may still be optimistic when reviews are templated or near-duplicates.
- Theme detection uses predefined keyword/phrase dictionaries and may miss semantically related language.
- Theme analysis is observational and does not establish causality.
- Model performance should be revalidated on an independent external dataset before operational use.

## Project Structure

```
NLP Sentiment Analysis/
├── src/
│   ├── data/
│   ├── preprocessing/
│   ├── models/
│   ├── insights/
│   │   ├── theme_analyzer.py
│   │   └── report_generator.py
│   └── visualization/
├── tests/
├── reports/
├── Customer_Reviews_Dataset.csv
├── README.md
└── requirements.txt
```
