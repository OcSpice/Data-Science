"""
Main Pipeline Module

Orchestrates the complete customer churn prediction pipeline.
Author: OGHENEOCHUKO EMMANUEL OGIDIAGBA
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from data.loader import DataLoader
from data.validator import DataValidator
from data.preprocessor import DataPreprocessor
from models.classifier import ChurnClassifier
from explainability.shap_explainer import SHAPExplainer
from business_impact.revenue_calculator import RevenueCalculator
from utils.report_generator import ReportGenerator


class ChurnPredictionPipeline:
    """
    Main pipeline orchestrating all components of the churn prediction system.
    
    Attributes:
        data_path: Path to the input dataset.
        output_dir: Directory for saving outputs.
        risk_threshold: Threshold for high-risk customer classification.
    """
    
    AUTHOR = "OGHENEOCHUKO EMMANUEL OGIDIAGBA"
    
    def __init__(self, data_path: str, output_dir: str = "outputs",
                 risk_threshold: float = 0.7):
        """
        Initialize the pipeline.
        
        Args:
            data_path: Path to the CSV dataset.
            output_dir: Directory for outputs.
            risk_threshold: Probability threshold for high-risk customers.
        """
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.risk_threshold = risk_threshold
        
        self.data_loader = DataLoader(data_path)
        self.validator = DataValidator()
        self.preprocessor = DataPreprocessor()
        self.classifier = ChurnClassifier()
        self.revenue_calculator = RevenueCalculator(risk_threshold=risk_threshold)
        self.report_generator = ReportGenerator(output_dir=str(self.output_dir))
        
        self.raw_data: Optional[pd.DataFrame] = None
        self.validation_report: Optional[Dict[str, Any]] = None
        self.model_metrics: Optional[Dict[str, float]] = None
        self.revenue_summary: Optional[Dict[str, Any]] = None
        self.shap_summary: Optional[Dict[str, Any]] = None
    
    def run(self, use_smote: bool = True, tune_params: bool = True,
            generate_shap: bool = True) -> Dict[str, Any]:
        """
        Execute the complete pipeline.
        
        Args:
            use_smote: Whether to apply SMOTE for class imbalance.
            tune_params: Whether to perform hyperparameter tuning.
            generate_shap: Whether to generate SHAP explanations.
            
        Returns:
            Dict[str, Any]: Complete pipeline results.
        """
        print("=" * 70)
        print("CUSTOMER CHURN PREDICTION PIPELINE")
        print(f"Author: {self.AUTHOR}")
        print("=" * 70)
        
        print("\n[1/7] Loading data...")
        self.raw_data = self.data_loader.load()
        print(f"      Loaded {len(self.raw_data)} records with {len(self.raw_data.columns)} columns")
        
        print("\n[2/7] Validating data quality...")
        is_valid, validation_report = self.validator.run_full_validation(self.raw_data)
        self.validation_report = validation_report
        print(f"      Validation status: {'PASSED' if is_valid else 'FAILED'}")
        print(f"      Warnings: {len(validation_report.get('warnings', []))}")
        
        anonymized_data = self.validator.anonymize_customer_id(self.raw_data)
        
        print("\n[3/7] Preprocessing data...")
        X_train, X_test, y_train, y_test = self.preprocessor.preprocess_full(anonymized_data)
        print(f"      Training set: {len(X_train)} samples")
        print(f"      Test set: {len(X_test)} samples")
        
        print("\n[4/7] Training model...")
        self.classifier.fit(X_train, y_train, use_smote=use_smote, tune_params=tune_params)
        print(f"      Best parameters: {self.classifier.best_params}")
        
        print("\n[5/7] Evaluating model performance...")
        self.model_metrics = self.classifier.evaluate(X_test, y_test)
        print(f"      Accuracy: {self.model_metrics['accuracy']:.4f}")
        print(f"      ROC-AUC: {self.model_metrics['roc_auc']:.4f}")
        print(f"      F1-Score: {self.model_metrics['f1_score']:.4f}")
        
        print("\n[6/7] Calculating at-risk revenue...")
        full_data_processed = self.preprocessor.encode_categorical_features(
            self.preprocessor.clean_data(anonymized_data), fit=False
        )
        X_full = self.preprocessor.prepare_features(full_data_processed)
        churn_probabilities = self.classifier.predict_proba(X_full)[:, 1]
        
        self.revenue_summary = self.revenue_calculator.calculate_at_risk_revenue(
            anonymized_data, churn_probabilities
        )
        print(f"      High-risk customers: {self.revenue_summary['high_risk_customers']}")
        print(f"      At-risk revenue: {self.revenue_summary['at_risk_revenue_formatted']}")
        
        shap_summary = {}
        if generate_shap:
            print("\n[7/7] Generating SHAP explanations...")
            shap_explainer = SHAPExplainer(self.classifier.model, X_train.sample(
                min(100, len(X_train)), random_state=42
            ))
            shap_explainer.compute_shap_values(X_test.sample(
                min(50, len(X_test)), random_state=42
            ))
            shap_summary = shap_explainer.generate_summary_report(
                str(self.output_dir / "shap_summary.json")
            )
            top_drivers = shap_explainer.get_top_features(n=5)
            print("      Top churn drivers identified:")
            for _, row in top_drivers.iterrows():
                print(f"        - {row['Feature']}: {row['MeanAbsoluteSHAP']:.4f}")
        
        self.shap_summary = shap_summary
        
        print("\n" + "=" * 70)
        print("GENERATING REPORTS")
        print("=" * 70)
        
        metrics_report = self.report_generator.generate_metrics_report(
            self.model_metrics,
            self.classifier.feature_importances,
            self.classifier.best_params
        )
        self.report_generator.save_json_report(metrics_report, "model_metrics.json")
        
        business_report = self.report_generator.generate_business_impact_report(
            self.revenue_summary, shap_summary
        )
        self.report_generator.save_json_report(business_report, "business_impact.json")
        
        full_report_path = self.report_generator.generate_full_pipeline_report(
            self.model_metrics,
            self.classifier.feature_importances,
            self.classifier.best_params,
            self.revenue_summary,
            shap_summary,
            self.validation_report
        )
        
        self.classifier.save_model(str(self.output_dir / "churn_model.joblib"))
        
        executive_summary = self.revenue_calculator.generate_executive_summary()
        summary_path = self.output_dir / "executive_summary.txt"
        with open(summary_path, "w") as f:
            f.write(executive_summary)
        
        print("\n" + "=" * 70)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print(f"\nOutputs saved to: {self.output_dir.absolute()}")
        print("Generated files:")
        for report in self.report_generator.reports_generated:
            print(f"  - {report}")
        print(f"  - {self.output_dir / 'churn_model.joblib'}")
        print(f"  - {summary_path}")
        
        return {
            "model_metrics": self.model_metrics,
            "revenue_summary": self.revenue_summary,
            "shap_summary": shap_summary,
            "validation_report": self.validation_report,
            "reports_generated": self.report_generator.reports_generated
        }


def main():
    """Main entry point for running the pipeline."""
    base_dir = Path(__file__).parent.parent
    data_path = base_dir / "Churn_Dataset.csv"
    output_dir = base_dir / "outputs"
    
    if not data_path.exists():
        print(f"Error: Dataset not found at {data_path}")
        sys.exit(1)
    
    pipeline = ChurnPredictionPipeline(
        data_path=str(data_path),
        output_dir=str(output_dir),
        risk_threshold=0.7
    )
    
    results = pipeline.run(
        use_smote=True,
        tune_params=True,
        generate_shap=True
    )
    
    print("\n" + "=" * 70)
    print("KEY RESULTS SUMMARY")
    print("=" * 70)
    print(f"Model Accuracy: {results['model_metrics']['accuracy']:.2%}")
    print(f"Model ROC-AUC: {results['model_metrics']['roc_auc']:.2%}")
    print(f"At-Risk Revenue: {results['revenue_summary']['at_risk_revenue_formatted']}")
    print(f"High-Risk Customers: {results['revenue_summary']['high_risk_customers']}")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    main()
