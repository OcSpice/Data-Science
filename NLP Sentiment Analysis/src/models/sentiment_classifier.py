"""
Sentiment classification and evaluation utilities.

The classifier module deliberately keeps model fitting separate from TF-IDF
fitting so the caller can fit text features on training data only.
"""

import joblib
import numpy as np
from typing import Dict

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


class SentimentClassifier:
    """Train and evaluate a classical multiclass sentiment classifier."""

    LABELS = ["Negative", "Neutral", "Positive"]
    label_mapping = {"Negative": 0, "Neutral": 1, "Positive": 2}
    inverse_label_mapping = {v: k for k, v in label_mapping.items()}

    def __init__(self, model_type: str = "svm", random_state: int = 42):
        self.model_type = model_type
        self.random_state = random_state
        self.is_fitted = False
        if model_type == "svm":
            self.model = LinearSVC(C=1.0, class_weight="balanced", max_iter=10000, random_state=random_state)
        elif model_type == "naive_bayes":
            self.model = MultinomialNB(alpha=0.1)
        elif model_type == "logistic_regression":
            self.model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=random_state)
        else:
            raise ValueError("Unsupported model type: svm, naive_bayes, or logistic_regression")

    def prepare_labels(self, labels) -> np.ndarray:
        return np.asarray([self.label_mapping[label] for label in labels])

    def inverse_transform_labels(self, numeric_labels) -> np.ndarray:
        return np.asarray([self.inverse_label_mapping[int(label)] for label in numeric_labels])

    def train(self, X, y, test_size: float = 0.2) -> Dict:
        """Backward-compatible helper that splits already-created features."""
        from sklearn.model_selection import train_test_split
        y_numeric = self.prepare_labels(y)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_numeric, test_size=test_size, random_state=self.random_state, stratify=y_numeric
        )
        return self.fit_and_evaluate(X_train, X_test, y_train, y_test)

    def fit_and_evaluate(self, X_train, X_test, y_train, y_test) -> Dict:
        """Fit on training features and evaluate only on held-out features."""
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        y_pred = self.model.predict(X_test)
        scores = self._decision_scores(X_test)
        y_true_binary = np.eye(len(self.LABELS))[y_test]

        return {
            "model": self.model_type,
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
            "roc_auc_macro": float(roc_auc_score(y_true_binary, scores, multi_class="ovr", average="macro")),
            "pr_auc_macro": float(average_precision_score(y_true_binary, scores, average="macro")),
            "train_size": int(len(y_train)),
            "test_size": int(len(y_test)),
            "classification_report": classification_report(
                y_test, y_pred, target_names=self.LABELS, output_dict=True, zero_division=0
            ),
            "confusion_matrix": confusion_matrix(y_test, y_pred, labels=[0, 1, 2]).tolist(),
        }

    def _decision_scores(self, X):
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        decision = np.asarray(self.model.decision_function(X))
        if decision.ndim == 1:
            decision = np.column_stack([-decision, np.zeros_like(decision), decision])
        exp_scores = np.exp(decision - decision.max(axis=1, keepdims=True))
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)

    def predict(self, X) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction.")
        return self.inverse_transform_labels(self.model.predict(X))

    def evaluate(self, X, y_true) -> str:
        if not self.is_fitted:
            raise ValueError("Model must be fitted first.")
        return classification_report(y_true, self.predict(X))

    def get_confusion_matrix(self, X, y_true) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted first.")
        return confusion_matrix(self.prepare_labels(y_true), self.model.predict(X), labels=[0, 1, 2])

    def save_model(self, filepath: str) -> None:
        if not self.is_fitted:
            raise ValueError("No model to save. Train the model first.")
        joblib.dump({"model": self.model, "model_type": self.model_type,
                     "label_mapping": self.label_mapping,
                     "inverse_label_mapping": self.inverse_label_mapping}, filepath)

    def load_model(self, filepath: str) -> "SentimentClassifier":
        data = joblib.load(filepath)
        self.model = data["model"]
        self.model_type = data["model_type"]
        self.label_mapping = data["label_mapping"]
        self.inverse_label_mapping = data["inverse_label_mapping"]
        self.is_fitted = True
        return self
