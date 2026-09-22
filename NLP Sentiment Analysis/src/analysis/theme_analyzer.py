"""Descriptive theme analysis for customer review text.

This module identifies recurring terms in negative reviews. It is descriptive
only: frequent terms are not treated as causal root causes.
"""

from typing import Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class FeedbackThemeAnalyzer:
    """Extract recurring lexical themes from review text."""

    def __init__(self, min_df: int = 5, max_features: int = 100):
        self.min_df = min_df
        self.max_features = max_features

    def top_terms(self, texts: List[str], top_n: int = 20) -> List[Dict]:
        if not texts:
            return []

        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=self.min_df,
            max_features=self.max_features,
            sublinear_tf=True,
        )
        matrix = vectorizer.fit_transform(texts)
        mean_scores = np.asarray(matrix.mean(axis=0)).ravel()
        terms = vectorizer.get_feature_names_out()
        order = np.argsort(mean_scores)[::-1][:top_n]

        return [
            {"term": str(terms[i]), "mean_tfidf": float(mean_scores[i])}
            for i in order
        ]

    def analyze(self, texts: List[str], labels: List[str], top_n: int = 20) -> Dict:
        negative_texts = [
            text for text, label in zip(texts, labels) if label == "Negative"
        ]
        return {
            "negative_review_count": len(negative_texts),
            "top_negative_terms": self.top_terms(negative_texts, top_n=top_n),
        }
