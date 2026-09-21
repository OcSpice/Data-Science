"""
NLP Sentiment Analysis and Customer Feedback Insight Engine
Author: OGHENEOCHUKU EMMANUEL OGIDIAGBA

Unit tests for core pipeline components.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data.data_loader import DataLoader
from preprocessing.text_preprocessor import TextPreprocessor
from models.tfidf_vectorizer import CustomTFIDFVectorizer
from models.sentiment_classifier import SentimentClassifier
from insights.root_cause_analyzer import RootCauseAnalyzer


class TestTextPreprocessor:
    """Tests for text preprocessing functionality."""
    
    def setup_method(self):
        self.preprocessor = TextPreprocessor()
    
    def test_clean_text_removes_html(self):
        text = "<p>This is a <b>test</b> review</p>"
        cleaned = self.preprocessor.clean_text(text)
        assert "<" not in cleaned
        assert ">" not in cleaned
    
    def test_clean_text_lowercases(self):
        text = "THIS IS A TEST"
        cleaned = self.preprocessor.clean_text(text)
        assert cleaned == cleaned.lower()
    
    def test_clean_text_removes_special_chars(self):
        text = "Test@#$% review!!!"
        cleaned = self.preprocessor.clean_text(text)
        assert "@" not in cleaned
        assert "#" not in cleaned
        assert "$" not in cleaned
    
    def test_anonymize_pii_emails(self):
        text = "Contact me at test@example.com"
        anonymized = self.preprocessor.anonymize_pii(text)
        assert "test@example.com" not in anonymized
        assert "[EMAIL_REDACTED]" in anonymized
    
    def test_anonymize_pii_phones(self):
        text = "Call 555-123-4567 for support"
        anonymized = self.preprocessor.anonymize_pii(text)
        assert "555-123-4567" not in anonymized
        assert "[PHONE_REDACTED]" in anonymized
    
    def test_tokenize(self):
        text = "this is a test"
        tokens = self.preprocessor.tokenize(text)
        assert len(tokens) == 4
        assert tokens == ['this', 'is', 'a', 'test']
    
    def test_remove_stopwords(self):
        tokens = ['this', 'is', 'a', 'test']
        filtered = self.preprocessor.remove_stopwords(tokens)
        assert 'test' in filtered
        assert 'this' not in filtered
    
    def test_lemmatize(self):
        tokens = ['running', 'studies', 'products']
        lemmatized = self.preprocessor.lemmatize(tokens)
        assert lemmatized[0] == 'running' or lemmatized[0] == 'run'
        assert lemmatized[1] == 'study' or lemmatized[1] == 'studies'
    
    def test_full_preprocess(self):
        text = "This is a GREAT product!!! <html> Contact: test@email.com"
        result = self.preprocessor.preprocess(text)
        assert isinstance(result, str)
        assert len(result) > 0


class TestTFIDFVectorizer:
    """Tests for TF-IDF vectorization."""
    
    def setup_method(self):
        self.vectorizer = CustomTFIDFVectorizer(max_features=100, min_df=1)
    
    def test_fit_transform(self):
        texts = ["great product", "bad service", "excellent quality"]
        X = self.vectorizer.fit_transform(texts)
        assert X.shape[0] == 3
        assert X.shape[1] <= 100
    
    def test_transform_without_fit_raises_error(self):
        texts = ["test text"]
        with pytest.raises(ValueError):
            self.vectorizer.transform(texts)
    
    def test_get_feature_names(self):
        texts = ["great product", "bad service"]
        self.vectorizer.fit_transform(texts)
        features = self.vectorizer.get_feature_names()
        assert len(features) > 0
        assert 'great' in features or 'product' in features


class TestSentimentClassifier:
    """Tests for sentiment classification model."""
    
    def setup_method(self):
        self.classifier = SentimentClassifier(model_type='svm')
    
    def test_train_and_predict(self):
        X = np.array([[0.1, 0.2], [0.8, 0.9], [0.3, 0.4], [0.9, 0.8], 
                      [0.2, 0.3], [0.7, 0.85]])
        y = ['Negative', 'Positive', 'Negative', 'Positive', 'Negative', 'Positive']
        
        metrics = self.classifier.train(X, y, test_size=0.33)
        assert 'accuracy' in metrics
        assert 0 <= metrics['accuracy'] <= 1
    
    def test_predict_without_train_raises_error(self):
        X = np.array([[0.1, 0.2]])
        with pytest.raises(ValueError):
            self.classifier.predict(X)
    
    def test_label_mapping(self):
        labels = ['Negative', 'Neutral', 'Positive']
        numeric = self.classifier.prepare_labels(labels)
        assert numeric[0] == 0
        assert numeric[1] == 1
        assert numeric[2] == 2
    
    def test_inverse_label_mapping(self):
        numeric = [0, 1, 2]
        labels = self.classifier.inverse_transform_labels(numeric)
        assert labels[0] == 'Negative'
        assert labels[1] == 'Neutral'
        assert labels[2] == 'Positive'


class TestRootCauseAnalyzer:
    """Tests for root-cause analysis functionality."""
    
    def setup_method(self):
        self.analyzer = RootCauseAnalyzer()
    
    def test_extract_ngrams_unigrams(self):
        text = "customer support is terrible"
        unigrams = self.analyzer.extract_ngrams(text, n=1)
        assert 'customer' in unigrams
        assert 'support' in unigrams
    
    def test_extract_ngrams_bigrams(self):
        text = "customer support is terrible"
        bigrams = self.analyzer.extract_ngrams(text, n=2)
        assert 'customer support' in bigrams
        assert 'support is' in bigrams
    
    def test_detect_customer_support_mentions(self):
        texts = [
            "customer support was unhelpful",
            "great product overall",
            "terrible customer service experience"
        ]
        count, percentage = self.analyzer.detect_customer_support_mentions(texts)
        assert count >= 2
        assert percentage >= 60
    
    def test_get_root_cause_summary(self):
        negative_texts = [
            "customer support never responded",
            "terrible service quality",
            "product broke after one day"
        ]
        summary = self.analyzer.get_root_cause_summary(negative_texts, len(negative_texts))
        
        assert 'total_negative_reviews' in summary
        assert 'reviews_mentioning_support' in summary
        assert 'top_root_cause_keywords' in summary
        assert summary['author'] == RootCauseAnalyzer.AUTHOR


class TestDataLoader:
    """Tests for data loading functionality."""
    
    def test_expected_columns(self):
        expected = DataLoader.EXPECTED_COLUMNS
        assert 'ReviewID' in expected
        assert 'Sentiment' in expected
        assert 'Review' in expected
    
    def test_valid_sentiments(self):
        valid = DataLoader.VALID_SENTIMENTS
        assert 'Positive' in valid
        assert 'Negative' in valid
        assert 'Neutral' in valid


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
