"""
Churn Classifier Module

Implements advanced machine learning models for churn prediction.
Author: OGHENEOCHUKO EMMANUEL OGIDIAGBA
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE


class ChurnClassifier:
    """
    Advanced churn prediction classifier with hyperparameter tuning.
    
    Uses Gradient Boosting with class weight balancing and optional SMOTE.
    
    Attributes:
        model: The trained classifier model.
        best_params: Best hyperparameters from grid search.
        metrics: Dictionary of evaluation metrics.
    """
    
    AUTHOR = "OGHENEOCHUKO EMMANUEL OGIDIAGBA"
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the ChurnClassifier.
        
        Args:
            random_state: Random seed for reproducibility.
        """
        self.random_state = random_state
        self.model: Optional[GradientBoostingClassifier] = None
        self.best_params: Dict[str, Any] = {}
        self.metrics: Dict[str, float] = {}
        self.feature_importances: Dict[str, float] = {}
        self._is_trained = False
    
    def _create_base_model(self, **kwargs) -> GradientBoostingClassifier:
        """Create base Gradient Boosting model."""
        return GradientBoostingClassifier(
            random_state=self.random_state,
            **kwargs
        )
    
    def tune_hyperparameters(self, X_train: pd.DataFrame, 
                              y_train: pd.Series,
                              use_smote: bool = True) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Args:
            X_train: Training features.
            y_train: Training target.
            use_smote: Whether to apply SMOTE for class imbalance.
            
        Returns:
            Dict[str, Any]: Best hyperparameters.
        """
        if use_smote and y_train.value_counts().min() > 5:
            smote = SMOTE(random_state=self.random_state)
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        else:
            X_resampled, y_resampled = X_train, y_train
        
        param_grid = {
            "n_estimators": [100],
            "max_depth": [5, 7],
            "learning_rate": [0.1],
            "min_samples_split": [2]
        }
        
        base_model = self._create_base_model()
        
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
        
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_resampled, y_resampled)
        
        self.best_params = grid_search.best_params_
        
        return self.best_params
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            use_smote: bool = True, tune_params: bool = True) -> "ChurnClassifier":
        """
        Train the classifier.
        
        Args:
            X_train: Training features.
            y_train: Training target.
            use_smote: Whether to apply SMOTE.
            tune_params: Whether to perform hyperparameter tuning.
            
        Returns:
            Self for method chaining.
        """
        if tune_params:
            self.tune_hyperparameters(X_train, y_train, use_smote)
        
        if use_smote and y_train.value_counts().min() > 5:
            smote = SMOTE(random_state=self.random_state)
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        else:
            X_resampled, y_resampled = X_train, y_train
        
        final_params = self.best_params if self.best_params else {}
        self.model = self._create_base_model(**final_params)
        self.model.fit(X_resampled, y_resampled)
        
        self.feature_importances = dict(
            zip(X_train.columns, self.model.feature_importances_)
        )
        
        self._is_trained = True
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Feature matrix.
            
        Returns:
            Array of predicted classes.
        """
        if not self._is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make probability predictions.
        
        Args:
            X: Feature matrix.
            
        Returns:
            Array of predicted probabilities for each class.
        """
        if not self._is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test: pd.DataFrame, 
                 y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            X_test: Test features.
            y_test: Test target.
            
        Returns:
            Dict[str, float]: Dictionary of evaluation metrics.
        """
        if not self._is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)[:, 1]
        
        self.metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_pred_proba)
        }
        
        conf_matrix = confusion_matrix(y_test, y_pred)
        self.metrics["true_negatives"] = int(conf_matrix[0, 0])
        self.metrics["false_positives"] = int(conf_matrix[0, 1])
        self.metrics["false_negatives"] = int(conf_matrix[1, 0])
        self.metrics["true_positives"] = int(conf_matrix[1, 1])
        
        return self.metrics
    
    def get_classification_report(self, X_test: pd.DataFrame,
                                   y_test: pd.Series) -> str:
        """
        Generate detailed classification report.
        
        Args:
            X_test: Test features.
            y_test: Test target.
            
        Returns:
            str: Formatted classification report.
        """
        y_pred = self.predict(X_test)
        report = classification_report(y_test, y_pred)
        return report
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model.
        """
        if not self._is_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            "model": self.model,
            "best_params": self.best_params,
            "feature_importances": self.feature_importances,
            "author": self.AUTHOR
        }
        
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath: str) -> "ChurnClassifier":
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model.
            
        Returns:
            Self for method chaining.
        """
        model_data = joblib.load(filepath)
        self.model = model_data["model"]
        self.best_params = model_data["best_params"]
        self.feature_importances = model_data.get("feature_importances", {})
        self._is_trained = True
        
        return self
    
    def get_feature_importance_ranking(self) -> pd.DataFrame:
        """
        Get feature importance ranking.
        
        Returns:
            pd.DataFrame: DataFrame with features ranked by importance.
        """
        if not self.feature_importances:
            raise ValueError("Model must be trained to get feature importances")
        
        df_importance = pd.DataFrame(
            list(self.feature_importances.items()),
            columns=["Feature", "Importance"]
        ).sort_values("Importance", ascending=False).reset_index(drop=True)
        
        return df_importance
