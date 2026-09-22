"""Report generation for the churn and retention economics pipeline."""
import json
from datetime import datetime,timezone
from pathlib import Path

AUTHOR="OGHENEOCHUKO EMMANUEL OGIDIAGBA"
PROJECT="Customer Churn Prediction & Retention Economics"

class ReportGenerator:
    def __init__(self,output_dir="outputs"):
        self.output_dir=Path(output_dir); self.output_dir.mkdir(parents=True,exist_ok=True); self.reports_generated=[]

    def save_json_report(self,report,filename):
        path=self.output_dir/filename
        path.write_text(json.dumps(report,indent=2,default=str))
        self.reports_generated.append(str(path))
        return str(path)

    def generate_metrics_report(self,model_metrics,best_params,threshold_analysis=None,scenario_analysis=None):
        return {
            "author":AUTHOR,"project":PROJECT,
            "generated_at":datetime.now(timezone.utc).isoformat(),
            "model_comparison":model_metrics,
            "random_forest_hyperparameters":best_params,
            "threshold_analysis":threshold_analysis or [],
            "retention_scenarios":scenario_analysis or []
        }

    def generate_full_pipeline_report(self,data_quality,metrics,revenue,shap,thresholds,scenarios):
        return self.save_json_report({
            "author":AUTHOR,"project":PROJECT,"version":"2.0.0",
            "generated_at":datetime.now(timezone.utc).isoformat(),
            "data_quality":data_quality,
            "model_comparison":metrics,
            "revenue_exposure":revenue,
            "explainability":shap,
            "threshold_analysis":thresholds,
            "retention_scenarios":scenarios,
            "limitations":[
                "Revenue exposure is not realized revenue loss.",
                "Retention success rates and intervention costs are scenario assumptions.",
                "SHAP explains model behavior and should not be interpreted as causal evidence.",
                "This is a portfolio-grade analytical prototype, not a production deployment."
            ]
        },"full_pipeline_report.json")

    def get_author_metadata(self):
        return {"author":AUTHOR,"project":PROJECT}
