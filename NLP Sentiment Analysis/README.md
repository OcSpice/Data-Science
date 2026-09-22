# NLP Sentiment Analysis and Customer Feedback Insight Engine

**Author:** OGHENEOCHUKU EMMANUEL OGIDIAGBA  
**Portfolio Category:** Data Science

## Project Overview
This project builds a leakage-aware NLP pipeline for classifying sentiment in customer reviews from the **Women's Clothing E-Commerce Reviews** dataset.

The workflow covers raw-data auditing, missing-review removal, duplicate-review detection, rating-based proxy labels, deterministic preprocessing, stratified splitting, training-only TF-IDF, four-model comparison, classification metrics, and descriptive negative-review theme analysis.

## Dataset and Label Methodology
The raw dataset contains **23,486 rows** and 11 columns.

Sentiment is derived from rating:

| Rating | Proxy sentiment |
|---|---|
| 1–2 | Negative |
| 3 | Neutral |
| 4–5 | Positive |

This is a **proxy-label methodology**. The classes represent rating-derived sentiment rather than independently human-annotated sentiment.

### Data-quality audit
- **845** missing `Review Text` values
- **0** exact duplicate rows
- **7** repeated non-empty review texts after normalization
- **22,634** unique reviews remaining for modeling

| Sentiment | Reviews | Approx. share |
|---|---:|---:|
| Negative | 2,369 | 10.5% |
| Neutral | 2,823 | 12.5% |
| Positive | 17,442 | 77.1% |

Because of this imbalance, **Macro-F1** is the primary comparison metric.

## Leakage Controls
The primary classifier is text-only. `Rating` is used to construct the target, while `Recommended IND` and `Positive Feedback Count` are excluded because they can encode post-review or target-related information. Demographic and product metadata are also excluded from the primary text experiment.

The split is stratified with random state 42. TF-IDF is fitted only on training text and then applied to the test set.

## Text Processing
- removes HTML and URLs
- anonymizes emails and phone numbers
- lowercases text
- removes non-alphabetic noise while retaining apostrophes
- normalizes whitespace
- combines review title and review text when a title exists

No NLTK download step is required by the rebuilt pipeline.

## Model Comparison
Models evaluated:
- Majority-class baseline
- Multinomial Naive Bayes
- Logistic Regression with balanced class weights
- Linear SVM with balanced class weights

| Model | Accuracy | Macro-F1 | Macro Precision | Macro Recall | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Majority baseline | 77.05% | 0.2901 | 0.2568 | 0.3333 | — |
| Naive Bayes | **82.37%** | 0.5692 | **0.6622** | 0.5304 | **0.9036** |
| Logistic Regression | 78.66% | **0.6100** | 0.5933 | **0.6392** | 0.8911 |
| Linear SVM | 80.80% | 0.6019 | 0.6025 | 0.6013 | 0.8763 |

The majority baseline is a reference point and is not treated as a ranking model for ROC/PR-AUC. Accuracy is not interpreted in isolation because of class imbalance.

## Descriptive Customer-Feedback Themes
Recurring terms in negative reviews are extracted using TF-IDF. Observed high-weight terms include **dress, like, fabric, fit, size, small, material, quality, color,** and **shirt**.

These are **descriptive textual themes**, not established causal root causes. Word frequency or TF-IDF association alone cannot establish causality.

## Project Structure
NLP Sentiment Analysis/
├── src/
│   ├── main_pipeline.py
│   ├── data/data_loader.py
│   ├── preprocessing/text_preprocessor.py
│   ├── models/tfidf_vectorizer.py
│   ├── models/sentiment_classifier.py
│   └── analysis/theme_analyzer.py
├── tests/test_pipeline.py
├── data/Womens Clothing E-Commerce Reviews.csv
├── reports/
├── README.md
└── requirements.txt

## Running the Project
From the project directory:

    python src/main_pipeline.py

Run tests with:

    pytest tests/test_pipeline.py -v

The pipeline writes evaluation output to `reports/model_evaluation.json`.

## Technical Highlights
- Python, Pandas, NumPy and scikit-learn
- TF-IDF unigram/bigram features
- Sparse-matrix processing
- Stratified evaluation
- Multiclass classification
- Macro-F1 evaluation
- One-vs-rest ROC-AUC and PR-AUC
- Descriptive NLP theme extraction
- Unit testing

## Limitations
1. Sentiment labels are derived from ratings rather than human sentiment annotations.
2. The positive class is substantially larger than the negative and neutral classes.
3. Review-level duplicate removal does not guarantee semantic near-duplicate removal.
4. TF-IDF themes indicate recurring language, not causal drivers.
5. Evaluation uses a single stratified holdout split; cross-validation could provide additional robustness.
6. Results should not be generalized beyond this dataset and collection context without further validation.

## Portfolio Value
This project demonstrates an end-to-end NLP workflow emphasizing **data quality, leakage prevention, class-imbalance awareness, model comparison, reproducibility, and evidence-based interpretation** rather than presenting a single accuracy number.

## Author
**OGHENEOCHUKU EMMANUEL OGIDIAGBA**

Data Science Portfolio