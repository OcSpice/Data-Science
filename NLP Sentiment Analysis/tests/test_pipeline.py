import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.data_loader import DataLoader
from models.tfidf_vectorizer import CustomTFIDFVectorizer
from models.sentiment_classifier import SentimentClassifier
from insights.theme_analyzer import ThemeAnalyzer
from preprocessing.text_preprocessor import TextPreprocessor


def test_preprocessor_removes_html_and_pii():
    p = TextPreprocessor()
    result = p.preprocess("GREAT product <b>test</b> test@example.com")
    assert "<" not in result
    assert "test@example.com" not in result


def test_tfidf_requires_fit():
    v = CustomTFIDFVectorizer(max_features=100, min_df=1)
    with pytest.raises(ValueError):
        v.transform(["test text"])


def test_tfidf_train_then_test_transform():
    v = CustomTFIDFVectorizer(max_features=100, min_df=1)
    X_train = v.fit_transform(["great product", "bad service"])
    X_test = v.transform(["great service"])
    assert X_train.shape[0] == 2
    assert X_test.shape[0] == 1


def test_classifier_computes_multiclass_metrics():
    X = np.array([
        [0.9, 0.1], [0.8, 0.2], [0.1, 0.9],
        [0.2, 0.8], [0.6, 0.4], [0.4, 0.6],
        [0.7, 0.3], [0.3, 0.7], [0.5, 0.5],
    ])
    y = ["Negative", "Negative", "Positive", "Positive", "Neutral",
         "Neutral", "Negative", "Positive", "Neutral"]
    model = SentimentClassifier("logistic_regression")
    metrics = model.train(X, y, test_size=1/3)
    for key in ["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"]:
        assert key in metrics
        assert 0 <= metrics[key] <= 1


def test_classifier_supports_naive_bayes_and_svm():
    X = np.array([
        [1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9], [0.5, 0.5], [0.4, 0.6],
    ])
    y = ["Negative", "Negative", "Positive", "Positive", "Neutral", "Neutral"]
    for model_type in ["naive_bayes", "svm"]:
        metrics = SentimentClassifier(model_type).train(X, y, test_size=0.5)
        assert "f1_macro" in metrics


def test_theme_analysis_is_descriptive():
    analyzer = ThemeAnalyzer()
    reviews = [
        "Customer support never responded",
        "The price is too expensive",
        "Great product quality",
        "Customer service was poor",
    ]
    sentiments = ["Negative", "Negative", "Positive", "Negative"]
    categories = ["FinTech", "Analytics", "Analytics", "FinTech"]
    result = analyzer.analyze_themes(reviews, sentiments, categories)
    support = next(x for x in result["themes"] if x["theme"] == "Customer Support")
    assert support["review_count"] == 2
    assert support["negative_pct_within_theme"] == 100.0
    assert "causal" in result["interpretation_note"].lower()


def test_duplicate_summary():
    loader = DataLoader("unused.csv")
    import pandas as pd
    loader.data = pd.DataFrame({
        "ReviewID": [1, 2, 3],
        "Product": ["A", "A", "B"],
        "Category": ["X", "X", "Y"],
        "Source": ["Web"] * 3,
        "Country": ["NG"] * 3,
        "Rating": [1, 2, 5],
        "Sentiment": ["Negative", "Negative", "Positive"],
        "Review": ["same text", "same text", "different"],
        "WordCount": [2, 2, 1],
        "CharCount": [9, 9, 9],
        "Topic": ["a", "a", "b"],
    })
    summary = loader.get_duplicate_summary()
    assert summary["exact_duplicate_rows"] == 2
    assert summary["unique_duplicate_texts"] == 1
    assert summary["duplicate_texts_with_conflicting_labels"] == 0


def test_expected_schema():
    assert "Sentiment" in DataLoader.EXPECTED_COLUMNS
    assert DataLoader.VALID_SENTIMENTS == {"Positive", "Negative", "Neutral"}
