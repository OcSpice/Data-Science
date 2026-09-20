"""
SHAP Explainer Module

Provides model explainability using SHAP (SHapley Additive exPlanations).
Author: OGHENEOCHUKO EMMANUEL OGIDIAGBA
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import shap


class SHAPExplainer:
    """
    Generates SHAP values for model interpretability.
    
    Explains why the model predicts certain customers will churn.
    
    Attributes:
        explainer: SHAP explainer object.
        shap_values: Computed SHAP values.
        feature_names: Names of features used in explanation.
    """
    
    AUTHOR = "OGHENEOCHUKO EMMANUEL OGIDIAGBA"
    
    def __init__(self, model, background_data: pd.DataFrame):
        """
        Initialize the SHAPExplainer.
        
        Args:
            model: Trained model to explain.
            background_data: Background dataset for SHAP value computation.
        """
        self.model = model
        self.background_data = background_data
        self.feature_names = list(background_data.columns)
        self.explainer: Optional[shap.Explainer] = None
        self.shap_values: Optional[np.ndarray] = None
        self.summary_data: Dict[str, Any] = {}
        
        self._fit()
    
    def _fit(self) -> None:
        """Fit the SHAP explainer on background data."""
        self.explainer = shap.Explainer(
            self.model.predict,
            self.background_data,
            feature_names=self.feature_names
        )
    
    def compute_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute SHAP values for given data.
        
        Args:
            X: Feature matrix to explain.
            
        Returns:
            Array of SHAP values.
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized")
        
        self.shap_values = self.explainer(X)
        
        return self.shap_values
    
    def get_top_features(self, n: int = 10) -> pd.DataFrame:
        """
        Get top N most important features based on mean absolute SHAP values.
        
        Args:
            n: Number of top features to return.
            
        Returns:
            pd.DataFrame: DataFrame with top features and their importance.
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed")
        
        mean_abs_shap = np.abs(self.shap_values.values).mean(axis=0)
        
        df_importance = pd.DataFrame({
            "Feature": self.feature_names,
            "MeanAbsoluteSHAP": mean_abs_shap
        }).sort_values("MeanAbsoluteSHAP", ascending=False).reset_index(drop=True)
        
        return df_importance.head(n)
    
    def explain_individual_prediction(self, X_sample: pd.DataFrame,
                                       index: int = 0) -> Dict[str, Any]:
        """
        Explain an individual prediction.
        
        Args:
            X_sample: Sample data to explain.
            index: Index of the sample to explain.
            
        Returns:
            Dict[str, Any]: Explanation dictionary with feature contributions.
        """
        if self.shap_values is None:
            single_explanation = self.explainer(X_sample.iloc[[index]])
        else:
            single_explanation = self.shap_values[index:index+1]
        
        feature_contributions = {}
        for i, feature in enumerate(self.feature_names):
            contribution = float(single_explanation.values[0, i])
            base_value = float(single_explanation.base_values[0])
            feature_contributions[feature] = {
                "contribution": contribution,
                "base_value": base_value,
                "feature_value": float(X_sample.iloc[index][feature])
            }
        
        return {
            "index": index,
            "predicted_probability": float(
                self.model.predict_proba(X_sample.iloc[[index]])[0, 1]
            ),
            "base_value": float(single_explanation.base_values[0]),
            "feature_contributions": feature_contributions
        }
    
    def generate_summary_report(self, output_path: str) -> Dict[str, Any]:
        """
        Generate a comprehensive SHAP summary report.
        
        Args:
            output_path: Path to save the JSON report.
            
        Returns:
            Dict[str, Any]: Summary report dictionary.
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed")
        
        top_features_df = self.get_top_features(n=15)
        
        self.summary_data = {
            "author": self.AUTHOR,
            "project": "Customer Churn Prediction Model",
            "total_features": len(self.feature_names),
            "top_churn_drivers": top_features_df.to_dict(orient="records"),
            "methodology": "SHAP (SHapley Additive exPlanations)",
            "interpretation_guide": {
                "positive_shap": "Increases probability of churn",
                "negative_shap": "Decreases probability of churn",
                "magnitude": "Strength of impact on prediction"
            },
            "key_business_insights": [
                "Month-to-month contracts are primary drivers of churn",
                "Fiber optic internet service correlates with higher churn",
                "Lack of TechSupport increases churn probability",
                "Shorter tenure customers show higher churn risk",
                "Paperless billing associated with increased churn"
            ]
        }
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(self.summary_data, f, indent=2)
        
        return self.summary_data
    
    def get_feature_direction(self, feature_name: str) -> str:
        """
        Determine if a feature generally increases or decreases churn.
        
        Args:
            feature_name: Name of the feature to analyze.
            
        Returns:
            str: Direction description ("increases", "decreases", or "mixed").
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed")
        
        if feature_name not in self.feature_names:
            raise ValueError(f"Feature {feature_name} not found")
        
        feature_idx = self.feature_names.index(feature_name)
        feature_shap = self.shap_values.values[:, feature_idx]
        
        mean_shap = np.mean(feature_shap)
        positive_ratio = np.sum(feature_shap > 0) / len(feature_shap)
        
        if mean_shap > 0.01:
            return "increases"
        elif mean_shap < -0.01:
            return "decreases"
        else:
            return "mixed"
