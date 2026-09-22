"""Leakage-safe multiclass sentiment model evaluation."""

from typing import Dict

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


LABELS = ["Negative", "Neutral", "Positive"]
LABEL_TO_INT = {label: i for i, label in enumerate(LABELS)}


class SentimentClassifier:
    """Train/evaluate baseline and linear NLP classifiers on TF-IDF features."""

    def __init__(self, model_type: str = "linear_svm", random_state: int = 42):
        self.model_type = model_type
        self.random_state = random_state
        self.is_fitted = False
        self.model = self._build_model(model_type)

    def _build_model(self, model_type):
        if model_type == "majority":
            return DummyClassifier(strategy="most_frequent")
        if model_type == "naive_bayes":
            return MultinomialNB(alpha=0.1)
        if model_type == "logistic_regression":
            return LogisticRegression(
                max_iter=2000, class_weight="balanced",
                random_state=self.random_state
            )
        if model_type == "linear_svm":
            return LinearSVC(
                C=1.0, class_weight="balanced",
                max_iter=10000, random_state=self.random_state
            )
        raise ValueError(f"Unsupported model type: {model_type}")

    def fit(self, X, y):
        y_encoded = np.asarray([LABEL_TO_INT[str(v)] for v in y])
        self.model.fit(X, y_encoded)
        self.is_fitted = True
        return self

    def predict_encoded(self, X):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction.")
        return self.model.predict(X)

    def predict(self, X):
        return np.asarray([LABELS[i] for i in self.predict_encoded(X)])

    def _scores(self, X):
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        if hasattr(self.model, "decision_function"):
            return self.model.decision_function(X)
        return None

    def evaluate(self, X, y_true) -> Dict[str, object]:
        y_encoded = np.asarray([LABEL_TO_INT[str(v)] for v in y_true])
        y_pred = self.predict_encoded(X)
        scores = self._scores(X)

        result = {
            "accuracy": float(accuracy_score(y_encoded, y_pred)),
            "macro_f1": float(f1_score(y_encoded, y_pred, average="macro")),
            "weighted_f1": float(f1_score(y_encoded, y_pred, average="weighted")),
            "macro_precision": float(precision_score(y_encoded, y_pred, average="macro", zero_division=0)),
            "macro_recall": float(recall_score(y_encoded, y_pred, average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(y_encoded, y_pred, labels=[0, 1, 2]).tolist(),
            "classification_report": classification_report(
                y_encoded, y_pred, labels=[0, 1, 2], target_names=LABELS,
                zero_division=0
            ),
        }

        if scores is not None:
            try:
                result["roc_auc_ovr_macro"] = float(
                    roc_auc_score(y_encoded, scores, multi_class="ovr", average="macro")
                )
                result["pr_auc_ovr_macro"] = float(
                    average_precision_score(
                        np.eye(len(LABELS))[y_encoded], scores, average="macro"
                    )
                )
            except ValueError:
                # Some model/test configurations may not support multiclass AUC.
                result["roc_auc_ovr_macro"] = None
                result["pr_auc_ovr_macro"] = None
        else:
            result["roc_auc_ovr_macro"] = None
            result["pr_auc_ovr_macro"] = None

        return result

    def save_model(self, filepath: str) -> None:
        import joblib
        if not self.is_fitted:
            raise ValueError("No model to save. Train the model first.")
        joblib.dump({
            "model": self.model,
            "model_type": self.model_type,
            "labels": LABELS,
        }, filepath)

    def load_model(self, filepath: str):
        import joblib
        data = joblib.load(filepath)
        self.model = data["model"]
        self.model_type = data["model_type"]
        self.is_fitted = True
        return self
