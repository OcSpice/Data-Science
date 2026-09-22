"""Tests for the rebuilt NLP sentiment analysis pipeline."""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import issparse

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.data_loader import DataLoader
from preprocessing.text_preprocessor import TextPreprocessor
from models.tfidf_vectorizer import CustomTFIDFVectorizer
from models.sentiment_classifier import SentimentClassifier, LABELS
from analysis.theme_analyzer import FeedbackThemeAnalyzer


class TestTextPreprocessor:
    def setup_method(self):
        self.preprocessor = TextPreprocessor()

    def test_clean_text_removes_html_and_lowercases(self):
        result = self.preprocessor.clean_text("<p>THIS is a TEST!</p>")
        assert result == "this is a test"

    def test_anonymize_pii(self):
        result = self.preprocessor.anonymize_pii("Email test@example.com or call 555-123-4567")
        assert "test@example.com" not in result
        assert "555-123-4567" not in result
        assert "EMAIL_REDACTED" in result
        assert "PHONE_REDACTED" in result

    def test_preprocess_batch(self):
        result = self.preprocessor.preprocess_batch(["GREAT product!", "Bad quality."])
        assert result == ["great product", "bad quality"]


class TestTFIDFVectorizer:
    def setup_method(self):
        self.vectorizer = CustomTFIDFVectorizer(max_features=100, min_df=1)

    def test_fit_transform_returns_sparse_matrix(self):
        X = self.vectorizer.fit_transform(["great product", "bad service", "excellent quality"])
        assert X.shape[0] == 3
        assert X.shape[1] <= 100
        assert issparse(X)

    def test_transform_without_fit_raises_error(self):
        with pytest.raises(ValueError):
            self.vectorizer.transform(["test text"])

    def test_test_only_term_is_not_learned(self):
        self.vectorizer.fit(["common word", "another common"])
        assert "testonlytoken" not in self.vectorizer.get_feature_names()
        X = self.vectorizer.transform(["testonlytoken common"])
        assert X.shape[1] == len(self.vectorizer.get_feature_names())


class TestSentimentClassifier:
    @pytest.mark.parametrize("model_type", ["majority", "naive_bayes", "logistic_regression", "linear_svm"])
    def test_train_and_evaluate(self, model_type):
        from sklearn.feature_extraction.text import TfidfVectorizer

        texts = [
            "terrible bad awful", "bad poor quality", "awful disappointing",
            "okay average product", "average not bad", "fine acceptable",
            "great excellent perfect", "excellent lovely", "great wonderful",
        ]
        labels = ["Negative", "Negative", "Negative", "Neutral", "Neutral", "Neutral",
                  "Positive", "Positive", "Positive"]
        X = TfidfVectorizer().fit_transform(texts)
        classifier = SentimentClassifier(model_type=model_type)
        classifier.fit(X, labels)
        metrics = classifier.evaluate(X, labels)
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["macro_f1"] <= 1
        assert len(metrics["confusion_matrix"]) == 3

    def test_predict_without_fit_raises_error(self):
        from scipy.sparse import csr_matrix

        classifier = SentimentClassifier("linear_svm")
        with pytest.raises(ValueError):
            classifier.predict(csr_matrix([[1.0, 0.0]]))

    def test_label_order(self):
        assert LABELS == ["Negative", "Neutral", "Positive"]


class TestDataLoader:
    def test_expected_columns_match_real_dataset_schema(self):
        expected = set(DataLoader.REQUIRED_COLUMNS)
        assert "Review Text" in expected
        assert "Rating" in expected
        assert "Recommended IND" in expected
        assert "Clothing ID" in expected

    def test_sentiment_map(self):
        assert DataLoader.SENTIMENT_MAP == {
            1: "Negative", 2: "Negative", 3: "Neutral", 4: "Positive", 5: "Positive"
        }

    def test_loader_removes_missing_and_duplicate_reviews(self, tmp_path):
        import pandas as pd

        source = pd.DataFrame({
            "Clothing ID": [1, 2, 3],
            "Age": [30, 40, 50],
            "Title": ["A", "B", "C"],
            "Review Text": ["Great fit", "", "Great fit"],
            "Rating": [5, 1, 5],
            "Recommended IND": [1, 0, 1],
            "Positive Feedback Count": [0, 1, 2],
            "Division Name": ["General"] * 3,
            "Department Name": ["Tops"] * 3,
            "Class Name": ["Blouses"] * 3,
        })
        path = tmp_path / "reviews.csv"
        source.to_csv(path, index=False)

        loader = DataLoader(str(path))
        result = loader.load()

        assert len(result) == 1
        assert result.iloc[0]["Sentiment"] == "Positive"
        assert loader.get_summary()["missing_review_text"] == 1
        assert loader.get_summary()["duplicate_review_rows_removed"] == 1


class TestFeedbackThemeAnalyzer:
    def test_negative_theme_analysis_is_descriptive(self):
        analyzer = FeedbackThemeAnalyzer(min_df=1)
        result = analyzer.analyze(
            ["bad fit and small size", "poor fabric quality", "great fit"],
            ["Negative", "Negative", "Positive"],
            top_n=5,
        )
        assert result["negative_review_count"] == 2
        assert result["top_negative_terms"]
        assert all("term" in item and "mean_tfidf" in item
                   for item in result["top_negative_terms"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
