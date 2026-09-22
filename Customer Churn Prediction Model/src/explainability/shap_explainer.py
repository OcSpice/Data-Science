"""TreeSHAP explanations for the Random Forest churn model."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import shap

class SHAPExplainer:
    def __init__(self,model,feature_names,background_data=None):
        self.model=model; self.feature_names=list(feature_names); self.background_data=background_data
        self.explainer=shap.TreeExplainer(model); self.shap_values=None; self.base_value=None; self.summary_data={}

    def compute_shap_values(self,X):
        e=self.explainer(X); values=e.values
        if values.ndim==3: values=values[:,:,1]
        self.shap_values=np.asarray(values)
        base=e.base_values
        if np.ndim(base)==2: base=base[:,1]
        self.base_value=float(np.mean(base))
        return self.shap_values

    @staticmethod
    def _source_feature(name):
        if name.startswith("num__"): return name.split("num__",1)[1]
        if name.startswith("cat__"): return name.split("cat__",1)[1].split("_",1)[0]
        return name

    def get_top_features(self,n=10):
        if self.shap_values is None: raise ValueError("SHAP values not computed.")
        raw=pd.DataFrame({"EncodedFeature":self.feature_names,"MeanAbsoluteSHAP":np.abs(self.shap_values).mean(axis=0)})
        raw["Feature"]=raw["EncodedFeature"].map(self._source_feature)
        return raw.groupby("Feature",as_index=False)["MeanAbsoluteSHAP"].sum().sort_values("MeanAbsoluteSHAP",ascending=False).head(n).reset_index(drop=True)

    def explain_individual_prediction(self,X,index=0):
        if self.shap_values is None: self.compute_shap_values(X)
        row=self.shap_values[index]
        contributions={f:{"contribution":float(v),"feature_value":float(X.iloc[index,i])} for i,(f,v) in enumerate(zip(self.feature_names,row))}
        return {"index":int(index),"predicted_probability":float(self.model.predict_proba(X.iloc[[index]])[0,1]),"base_value":self.base_value,"feature_contributions":contributions}

    def generate_summary_report(self,output_path):
        top=self.get_top_features(15)
        self.summary_data={
            "methodology":"SHAP TreeExplainer on held-out test data",
            "total_encoded_features":len(self.feature_names),
            "top_churn_drivers":top.to_dict(orient="records"),
            "interpretation_guide":{
                "positive_shap":"Moves model output toward higher churn probability.",
                "negative_shap":"Moves model output toward lower churn probability.",
                "magnitude":"Absolute SHAP magnitude indicates contribution strength."
            }
        }
        path=Path(output_path); path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(self.summary_data,indent=2))
        return self.summary_data
