"""
NLP Sentiment Analysis and Customer Feedback Insight Engine
Author: OGHENEOCHUKU EMMANUEL OGIDIAGBA

This module implements TF-IDF vectorization for text feature extraction.
"""

import numpy as np
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer


class CustomTFIDFVectorizer:
    """
    Custom TF-IDF vectorizer with configurable parameters.
    
    Attributes:
        vectorizer: Scikit-learn TfidfVectorizer instance
        max_features: Maximum number of features to extract
    """
    
    AUTHOR = "OGHENEOCHUKU EMMANUEL OGIDIAGBA"
    
    def __init__(self, max_features: int = 5000, ngram_range: Tuple[int, int] = (1, 2),
                 min_df: int = 2, max_df: float = 0.8):
        """
        Initialize the TF-IDF vectorizer.
        
        Args:
            max_features: Maximum number of features to keep
            ngram_range: Range of n-grams to consider
            min_df: Minimum document frequency threshold
            max_df: Maximum document frequency threshold
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            stop_words='english',
            sublinear_tf=True
        )
        self.is_fitted = False
    
    def fit(self, texts: List[str]) -> 'CustomTFIDFVectorizer':
        """
        Fit the vectorizer on a corpus of texts.
        
        Args:
            texts: List of preprocessed text documents
            
        Returns:
            Self for method chaining
        """
        self.vectorizer.fit(texts)
        self.is_fitted = True
        return self
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts into TF-IDF feature vectors.
        
        Args:
            texts: List of preprocessed text documents
            
        Returns:
            TF-IDF feature matrix
            
        Raises:
            ValueError: If vectorizer is not fitted
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before transform. Call fit() first.")
        
        return self.vectorizer.transform(texts).toarray()
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        Fit and transform texts in one step.
        
        Args:
            texts: List of preprocessed text documents
            
        Returns:
            TF-IDF feature matrix
        """
        self.is_fitted = True
        return self.vectorizer.fit_transform(texts).toarray()
    
    def get_feature_names(self) -> List[str]:
        """
        Get the feature names (vocabulary) from the fitted vectorizer.
        
        Returns:
            List of feature names
            
        Raises:
            ValueError: If vectorizer is not fitted
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted first.")
        
        return self.vectorizer.get_feature_names_out().tolist()
    
    def get_top_features_per_class(self, labels: np.ndarray, 
                                   texts: List[str], 
                                   top_n: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        """
        Get top TF-IDF features for each sentiment class.
        
        Args:
            labels: Array of sentiment labels
            texts: List of preprocessed texts
            top_n: Number of top features to return per class
            
        Returns:
            Dictionary mapping class labels to list of (feature, score) tuples
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted first.")
        
        tfidf_matrix = self.vectorizer.transform(texts)
        feature_names = self.get_feature_names()
        
        result = {}
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            class_mask = labels == label
            class_tfidf = tfidf_matrix[class_mask].mean(axis=0).A1
            top_indices = class_tfidf.argsort()[-top_n:][::-1]
            
            top_features = [(feature_names[i], float(class_tfidf[i])) 
                           for i in top_indices]
            result[str(label)] = top_features
        
        return result
