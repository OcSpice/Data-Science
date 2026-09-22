"""End-to-end customer churn prediction and retention economics pipeline."""
import sys
from pathlib import Path
import json
import pandas as pd

BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE))

from data.loader import DataLoader
from data.validator import DataValidator
from data.preprocessor import DataPreprocessor
from models.classifier import ChurnClassifier
from explainability.shap_explainer import SHAPExplainer
from business_impact.revenue_calculator import RevenueCalculator
from utils.report_generator import ReportGenerator

class ChurnPredictionPipeline:
    def __init__(self,data_path,output_dir="outputs",risk_threshold=0.7):
        self.data_path=data_path; self.output_dir=Path(output_dir); self.risk_threshold=risk_threshold
        self.loader=DataLoader(data_path); self.validator=DataValidator(); self.preprocessor=DataPreprocessor()
        self.classifier=ChurnClassifier(self.preprocessor); self.calculator=RevenueCalculator(risk_threshold); self.reporter=ReportGenerator(output_dir)

    def run(self,tune_params=True,generate_shap=True):
        print("CUSTOMER CHURN PREDICTION & RETENTION ECONOMICS")
        raw=self.loader.load()
        valid,validation=self.validator.run_full_validation(raw)
        if not valid: raise ValueError(f"Data validation failed: {validation.get('errors',[])}")
        anonymized=self.validator.anonymize_customer_id(raw)
        X_train,X_test,y_train,y_test=self.preprocessor.split_raw_data(anonymized)

        self.classifier.fit(X_train,y_train,tune_params=tune_params)
        metrics=self.classifier.evaluate(X_test,y_test)

        probabilities=self.classifier.predict_proba(X_test)
        revenue=self.calculator.calculate_at_risk_revenue(X_test,probabilities)
        thresholds=self.calculator.threshold_analysis(X_test,probabilities)
        scenarios=self.calculator.retention_scenarios(thresholds)

        threshold_records=thresholds.to_dict(orient="records")
        scenario_records=scenarios.to_dict(orient="records")
        shap_summary={}

        if generate_shap:
            X_transformed=self.preprocessor.transform(X_test,"tree")
            names=self.preprocessor.get_feature_names("tree")
            explainer=SHAPExplainer(self.classifier.get_model("random_forest"),names)
            sample=min(500,len(X_transformed))
            sample_array=X_transformed[:sample]
            explainer.compute_shap_values(sample_array)
            shap_summary=explainer.generate_summary_report(str(self.output_dir/"shap_summary.json"))
            local=explainer.explain_individual_prediction(pd.DataFrame(sample_array,columns=names),0)
            shap_summary["local_example"]=local
            (self.output_dir/"shap_summary.json").write_text(json.dumps(shap_summary,indent=2,default=str))

        self.reporter.save_json_report(
            self.reporter.generate_metrics_report(metrics,self.classifier.best_params,threshold_records,scenario_records),
            "model_metrics.json"
        )
        self.reporter.save_json_report({
            "revenue_exposure":revenue,
            "threshold_analysis":threshold_records,
            "retention_scenarios":scenario_records
        },"business_impact.json")
        self.reporter.generate_full_pipeline_report(validation,metrics,revenue,shap_summary,threshold_records,scenario_records)

        self.classifier.save_model(str(self.output_dir/"churn_model.joblib"))
        executive=(
            "CUSTOMER CHURN & RETENTION ECONOMICS - EXECUTIVE SUMMARY\n"
            "========================================================\n\n"
            f"Random Forest ROC-AUC: {metrics['random_forest']['roc_auc']:.3f}\n"
            f"Random Forest PR-AUC: {metrics['random_forest']['pr_auc']:.3f}\n"
            f"Random Forest Recall @ 0.50: {metrics['random_forest']['recall']:.3f}\n"
            f"High-risk threshold: {revenue['risk_threshold']:.2f}\n"
            f"High-risk customers: {revenue['high_risk_customers']:,}\n"
            f"Monthly revenue exposure: {revenue['at_risk_revenue_formatted']}\n\n"
            "Revenue exposure is a prioritization metric, not a forecast of realized loss. "
            "Retention scenarios use hypothetical intervention assumptions.\n"
        )
        (self.output_dir/"executive_summary.txt").write_text(executive)
        return {"model_metrics":metrics,"revenue_summary":revenue,"threshold_analysis":threshold_records,"retention_scenarios":scenario_records,"shap_summary":shap_summary,"validation_report":validation}

def main():
    root=BASE.parent
    data_path=root/"Churn_Dataset.csv"; output_dir=root/"outputs"
    if not data_path.exists(): raise SystemExit(f"Dataset not found: {data_path}")
    results=ChurnPredictionPipeline(data_path,output_dir,0.7).run(tune_params=True,generate_shap=True)
    print(json.dumps({"random_forest":results["model_metrics"]["random_forest"],"high_risk_customers":results["revenue_summary"]["high_risk_customers"],"monthly_revenue_exposure":results["revenue_summary"]["at_risk_revenue_formatted"]},indent=2))

if __name__=="__main__":
    main()
