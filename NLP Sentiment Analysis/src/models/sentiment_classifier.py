"""
NLP Sentiment Analysis and Customer Feedback Insight Engine
Author: OGHENEOCHUKU EMMANUEL OGIDIAGBA

This module implements sentiment classification models.
"""

import numpy as np
from typing import Tuple, Dict, Optional
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib


class SentimentClassifier:
    """
    Sentiment classification model with support for multiple algorithms.
    
    Attributes:
        model_type: Type of model to use ('svm' or 'naive_bayes')
        model: Trained classifier instance
    """
    
    AUTHOR = "OGHENEOCHUKU EMMANUEL OGIDIAGBA"
    
    def __init__(self, model_type: str = 'svm', random_state: int = 42):
        """
        Initialize the sentiment classifier.
        
        Args:
            model_type: Model algorithm ('svm' or 'naive_bayes')
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.is_fitted = False
        
        if model_type == 'svm':
            self.model = LinearSVC(
                C=1.0,
                class_weight='balanced',
                max_iter=10000,
                random_state=random_state
            )
        elif model_type == 'naive_bayes':
            self.model = MultinomialNB(alpha=0.1)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        self.label_mapping = {'Negative': 0, 'Neutral': 1, 'Positive': 2}
        self.inverse_label_mapping = {v: k for k, v in self.label_mapping.items()}
    
    def prepare_labels(self, labels: np.ndarray) -> np.ndarray:
        """
        Convert string labels to numeric labels.
        
        Args:
            labels: Array of string labels
            
        Returns:
            Array of numeric labels
        """
        return np.array([self.label_mapping.get(label, 1) for label in labels])
    
    def inverse_transform_labels(self, numeric_labels: np.ndarray) -> np.ndarray:
        """
        Convert numeric labels back to string labels.
        
        Args:
            numeric_labels: Array of numeric labels
            
        Returns:
            Array of string labels
        """
        return np.array([self.inverse_label_mapping.get(label, 'Neutral') 
                        for label in numeric_labels])
    
    def train(self, X: np.ndarray, y: np.ndarray, 
              test_size: float = 0.2) -> Dict[str, float]:
        """
        Train the sentiment classifier.
        
        Args:
            X: TF-IDF feature matrix
            y: Array of sentiment labels (strings)
            test_size: Proportion of data to use for testing
            
        Returns:
            Dictionary containing evaluation metrics
        """
        y_numeric = self.prepare_labels(y)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_numeric, test_size=test_size, 
            random_state=self.random_state, stratify=y_numeric
        )
        
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        y_pred = self.model.predict(X_test)
        
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'test_size': len(y_test),
            'train_size': len(y_train)
        }
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict sentiment for input features.
        
        Args:
            X: TF-IDF feature matrix
            
        Returns:
            Array of predicted sentiment labels (strings)
            
        Raises:
            ValueError: If model is not fitted
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction. Call train() first.")
        
        predictions = self.model.predict(X)
        return self.inverse_transform_labels(predictions)
    
    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> str:
        """
        Generate detailed evaluation report.
        
        Args:
            X: TF-IDF feature matrix
            y_true: True sentiment labels
            
        Returns:
            Classification report as string
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first.")
        
        y_pred = self.predict(X)
        return classification_report(y_true, y_pred)
    
    def get_confusion_matrix(self, X: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Compute confusion matrix.
        
        Args:
            X: TF-IDF feature matrix
            y_true: True sentiment labels
            
        Returns:
            Confusion matrix as numpy array
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first.")
        
        y_pred = self.predict(X)
        labels = ['Negative', 'Neutral', 'Positive']
        return confusion_matrix(y_true, y_pred, labels=labels)
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("No model to save. Train the model first.")
        
        joblib.dump({
            'model': self.model,
            'model_type': self.model_type,
            'label_mapping': self.label_mapping,
            'inverse_label_mapping': self.inverse_label_mapping
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> 'SentimentClassifier':
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Self with loaded model
        """
        data = joblib.load(filepath)
        self.model = data['model']
        self.model_type = data['model_type']
        self.label_mapping = data['label_mapping']
        self.inverse_label_mapping = data['inverse_label_mapping']
        self.is_fitted = True
        return self
