"""
Report Generator Module

Generates comprehensive reports with persistent metadata.
Author: OGHENEOCHUKO EMMANUEL OGIDIAGBA
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


class ReportGenerator:
    """
    Generates and saves comprehensive pipeline reports.
    
    Ensures author metadata persists across all generated outputs.
    
    Attributes:
        author: Author name constant for metadata persistence.
        project_name: Name of the project.
    """
    
    AUTHOR = "OGHENEOCHUKO EMMANUEL OGIDIAGBA"
    PROJECT_NAME = "Customer Churn Prediction Model"
    
    def __init__(self, output_dir: str = "outputs"):
        """
        Initialize the ReportGenerator.
        
        Args:
            output_dir: Directory to save reports.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reports_generated: list = []
    
    def generate_metrics_report(self, 
                                 model_metrics: Dict[str, float],
                                 feature_importance: Dict[str, float],
                                 best_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a comprehensive metrics report.
        
        Args:
            model_metrics: Dictionary of model evaluation metrics.
            feature_importance: Dictionary of feature importances.
            best_params: Best hyperparameters from tuning.
            
        Returns:
            Dict[str, Any]: Complete metrics report.
        """
        report = {
            "author": self.AUTHOR,
            "project": self.PROJECT_NAME,
            "generated_at": datetime.now().isoformat(),
            "model_performance": {
                "accuracy": model_metrics.get("accuracy", 0),
                "precision": model_metrics.get("precision", 0),
                "recall": model_metrics.get("recall", 0),
                "f1_score": model_metrics.get("f1_score", 0),
                "roc_auc": model_metrics.get("roc_auc", 0),
                "confusion_matrix": {
                    "true_negatives": model_metrics.get("true_negatives", 0),
                    "false_positives": model_metrics.get("false_positives", 0),
                    "false_negatives": model_metrics.get("false_negatives", 0),
                    "true_positives": model_metrics.get("true_positives", 0)
                }
            },
            "hyperparameters": best_params,
            "feature_importance_ranking": [
                {"feature": k, "importance": v}
                for k, v in sorted(
                    feature_importance.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
            ],
            "performance_thresholds": {
                "accuracy_target": 0.80,
                "roc_auc_target": 0.75,
                "accuracy_met": model_metrics.get("accuracy", 0) >= 0.80,
                "roc_auc_met": model_metrics.get("roc_auc", 0) >= 0.75
            }
        }
        
        return report
    
    def generate_business_impact_report(self,
                                         revenue_summary: Dict[str, Any],
                                         shap_summary: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a business impact report combining revenue and explainability.
        
        Args:
            revenue_summary: Revenue analysis summary.
            shap_summary: SHAP analysis summary.
            
        Returns:
            Dict[str, Any]: Combined business impact report.
        """
        report = {
            "author": self.AUTHOR,
            "project": self.PROJECT_NAME,
            "generated_at": datetime.now().isoformat(),
            "revenue_impact": {
                "at_risk_monthly": revenue_summary.get("at_risk_revenue_monthly", 0),
                "at_risk_formatted": revenue_summary.get("at_risk_revenue_formatted", "$0"),
                "projected_annual_loss": revenue_summary.get("projected_annual_loss", 0),
                "high_risk_customers": revenue_summary.get("high_risk_customers", 0),
                "key_finding": revenue_summary.get("key_finding", "")
            },
            "churn_drivers": shap_summary.get("top_churn_drivers", [])[:5],
            "business_insights": shap_summary.get("key_business_insights", []),
            "recommendations": revenue_summary.get("recommendations", [])
        }
        
        return report
    
    def save_json_report(self, report: Dict[str, Any], filename: str) -> str:
        """
        Save a report as JSON file.
        
        Args:
            report: Report dictionary to save.
            filename: Name of the output file.
            
        Returns:
            str: Path to saved file.
        """
        filepath = self.output_dir / filename
        
        with open(filepath, "w") as f:
            json.dump(report, f, indent=2)
        
        self.reports_generated.append(str(filepath))
        
        return str(filepath)
    
    def generate_full_pipeline_report(self,
                                       model_metrics: Dict[str, float],
                                       feature_importance: Dict[str, float],
                                       best_params: Dict[str, Any],
                                       revenue_summary: Dict[str, Any],
                                       shap_summary: Dict[str, Any],
                                       validation_report: Dict[str, Any]) -> str:
        """
        Generate a complete pipeline report.
        
        Args:
            model_metrics: Model evaluation metrics.
            feature_importance: Feature importance scores.
            best_params: Best hyperparameters.
            revenue_summary: Revenue analysis summary.
            shap_summary: SHAP analysis summary.
            validation_report: Data validation report.
            
        Returns:
            str: Path to saved report file.
        """
        full_report = {
            "author": self.AUTHOR,
            "project": self.PROJECT_NAME,
            "version": "1.0.0",
            "generated_at": datetime.now().isoformat(),
            "executive_summary": {
                "model_accuracy": f"{model_metrics.get('accuracy', 0):.2%}",
                "model_roc_auc": f"{model_metrics.get('roc_auc', 0):.2%}",
                "at_risk_revenue": revenue_summary.get("at_risk_revenue_formatted", "$0"),
                "high_risk_customer_count": revenue_summary.get("high_risk_customers", 0),
                "primary_churn_driver": shap_summary.get("top_churn_drivers", [{}])[0].get(
                    "Feature", "Unknown"
                ) if shap_summary.get("top_churn_drivers") else "Unknown"
            },
            "data_quality": validation_report,
            "model_performance": model_metrics,
            "hyperparameters": best_params,
            "feature_importance": feature_importance,
            "business_impact": {
                "revenue_analysis": revenue_summary,
                "explainability": shap_summary
            },
            "metadata": {
                "author": self.AUTHOR,
                "contact": "portfolio@example.com",
                "repository": "Customer Churn Prediction Model",
                "portfolio_category": "Data Science"
            }
        }
        
        filepath = self.save_json_report(full_report, "full_pipeline_report.json")
        
        return filepath
    
    def get_author_metadata(self) -> Dict[str, str]:
        """
        Get persistent author metadata.
        
        Returns:
            Dict[str, str]: Author metadata dictionary.
        """
        return {
            "author": self.AUTHOR,
            "project": self.PROJECT_NAME
        }
