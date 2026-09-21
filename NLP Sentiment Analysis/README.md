# NLP Sentiment Analysis and Customer Feedback Insight Engine

**Author:** OGHENEOCHUKU EMMANUEL OGIDIAGBA  
**Portfolio Category:** Data Science  
**Dataset Size:** 12,000 Customer Reviews

## Project Overview

This repository contains a production-ready Python pipeline for sentiment analysis and customer feedback insight extraction. The system processes unstructured text data from customer reviews to deliver actionable business insights and product improvement strategies.

### Key Features

- **Data Quality Engine:** Clean data loading with schema validation, text noise removal (HTML tags, special characters, extra whitespace), and PII anonymization
- **Custom NLP Pipeline:** Tokenization, stopword removal, lemmatization, and TF-IDF vectorization
- **Sentiment Classification:** SVM-based model for accurate sentiment prediction (Positive, Negative, Neutral)
- **Root-Cause Analysis:** Automated keyword extraction identifying why customers are unhappy
- **Automated Reporting:** Insight-driven visualizations and comprehensive JSON/text reports

## Key Business Discovery

**Primary Finding:** "Customer support" is the most frequently occurring keyword phrase in negative reviews, providing a direct, data-backed recommendation for business improvement.

## Directory Structure

```
NLP Sentiment Analysis/
├── src/
│   ├── __init__.py
│   ├── main_pipeline.py          # Main orchestration module
│   ├── data/
│   │   ├── __init__.py
│   │   └── data_loader.py        # Data loading and validation
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── text_preprocessor.py  # Text cleaning and PII anonymization
│   ├── models/
│   │   ├── __init__.py
│   │   ├── tfidf_vectorizer.py   # Custom TF-IDF implementation
│   │   └── sentiment_classifier.py # Sentiment classification model
│   ├── insights/
│   │   ├── __init__.py
│   │   ├── root_cause_analyzer.py # Keyword extraction and root-cause analysis
│   │   └── report_generator.py   # Automated report generation
│   └── visualization/
│       ├── __init__.py
│       └── plot_generator.py     # Visualization generation
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py          # Unit tests for all components
├── reports/                       # Generated reports and visualizations
├── configs/                       # Configuration files
├── Customer_Reviews_Dataset.csv   # Input dataset (12,000 reviews)
├── README.md                      # This file
└── requirements.txt               # Python dependencies
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Run Complete Pipeline

```bash
cd "NLP Sentiment Analysis"
python src/main_pipeline.py
```

### Programmatic Usage

```python
from src.main_pipeline import SentimentAnalysisPipeline

# Initialize pipeline
pipeline = SentimentAnalysisPipeline(
    data_path='Customer_Reviews_Dataset.csv',
    output_dir='reports'
)

# Execute full analysis
results = pipeline.run(generate_reports=True)

# Predict sentiment for new reviews
new_reviews = ["Great product!", "Terrible service"]
predictions = pipeline.predict_sentiment(new_reviews)
print(predictions)
```

### Run Unit Tests

```bash
pytest tests/test_pipeline.py -v
```

## Output Files

After running the pipeline, the following outputs are generated in the `reports/` directory:

1. **sentiment_distribution.png** - Bar chart showing sentiment breakdown
2. **negative_reviews_wordcloud.png** - Word cloud of negative review terms
3. **top_root_causes.png** - Top 10 keywords driving negative sentiment
4. **sentiment_analysis_report_*.json** - Comprehensive JSON report with all metrics
5. **sentiment_analysis_summary_*.txt** - Human-readable text summary
6. **pipeline_metadata.json** - Persistent metadata with author attribution

## Metrics and Results

### Dataset Statistics
- **Total Reviews:** 12,000
- **Sentiment Categories:** Positive, Negative, Neutral
- **Product Categories:** Multiple (Analytics, Integration, HR Tech, FinTech, etc.)

### Model Performance
- **Algorithm:** Support Vector Machine (SVM) with balanced class weights
- **Features:** TF-IDF with up to 3,000 features and bigrams
- **Evaluation:** Train/test split with stratification

### Root-Cause Analysis
The pipeline automatically identifies the top keywords and phrases associated with negative sentiment. The primary discovery is that **"customer support"** related issues are the leading cause of customer dissatisfaction.

## Business Recommendations

Based on the analysis:

1. **Expand customer support team capacity** to reduce response times
2. **Implement faster SLAs** for support ticket resolution
3. **Create self-service knowledge base** for common issues
4. **Establish proactive outreach** for customers with unresolved tickets
5. **Monitor support-related keywords** in real-time dashboards

## Technical Highlights

### Data Processing
- Schema validation ensures data integrity
- HTML tag removal and text normalization
- PII detection and anonymization (emails, phone numbers, names)

### NLP Pipeline
- NLTK-based tokenization and lemmatization
- Custom stopword filtering
- TF-IDF vectorization with configurable n-grams

### Machine Learning
- SVM classifier with balanced class weights
- Stratified train/test splitting
- Comprehensive evaluation metrics

### Reporting
- Author metadata persists across all generated files
- JSON reports for programmatic access
- Visual summaries for stakeholder presentations

## Author Information

**Name:** OGHENEOCHUKU EMMANUEL OGIDIAGBA  
**Role:** Senior Data Scientist / Software Engineer  
**Portfolio Track:** Data Science (with strong Data Analysis foundations)

This project demonstrates expertise in:
- Natural Language Processing (NLP)
- Text pipeline construction
- Automated root-cause extraction
- Translating unstructured data into business insights
- Building robust, reproducible data pipelines

## License

This project is part of a professional portfolio. All rights reserved.

---

*Generated by the NLP Sentiment Analysis and Customer Feedback Insight Engine*  
*Author: OGHENEOCHUKU EMMANUEL OGIDIAGBA*
