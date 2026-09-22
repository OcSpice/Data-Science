"""Retention economics and threshold analysis."""
import numpy as np
import pandas as pd

class RevenueCalculator:
    def __init__(self,risk_threshold=0.7,monthly_charges_column="MonthlyCharges"):
        if not 0<risk_threshold<1: raise ValueError("risk_threshold must be between 0 and 1.")
        self.risk_threshold=risk_threshold; self.monthly_charges_column=monthly_charges_column
        self.at_risk_customers=None; self.summary_report={}

    def _validate_inputs(self,df,p):
        if len(df)!=len(p): raise ValueError("Probability array length must match customer rows.")
        if self.monthly_charges_column not in df: raise ValueError("Required MonthlyCharges column not found.")
        p=np.asarray(p,dtype=float)
        if np.any((p<0)|(p>1)): raise ValueError("Churn probabilities must be between 0 and 1.")
        return p

    def calculate_at_risk_revenue(self,df,p):
        p=self._validate_inputs(df,p); out=df.copy(); out["ChurnProbability"]=p
        out["RiskSegment"]=np.select([p>=self.risk_threshold,(p>=0.4)&(p<self.risk_threshold)],["High Risk","Medium Risk"],default="Low Risk")
        self.at_risk_customers=out[out["RiskSegment"]=="High Risk"].copy()
        monthly=float(self.at_risk_customers[self.monthly_charges_column].sum()); count=len(self.at_risk_customers); total=len(out)
        self.summary_report={
            "analysis_type":"Revenue Exposure and Retention Economics",
            "risk_threshold":self.risk_threshold,
            "total_customers_analyzed":int(total),
            "high_risk_customers":int(count),
            "high_risk_percentage":round(count/total*100,2) if total else 0.0,
            "at_risk_revenue_monthly":round(monthly,2),
            "at_risk_revenue_annualized":round(monthly*12,2),
            "at_risk_revenue_formatted":"$"+"{:,.2f}".format(monthly),
            "revenue_exposure_definition":"MonthlyCharges associated with customers above the selected churn-probability threshold; not a forecast of realized revenue loss."
        }
        return self.summary_report

    def threshold_analysis(self,df,p,thresholds=(0.3,0.4,0.5,0.6,0.7,0.8)):
        p=self._validate_inputs(df,p); rows=[]
        for t in thresholds:
            mask=p>=t; target=df.loc[mask,self.monthly_charges_column]
            rows.append({"threshold":float(t),"customers_targeted":int(mask.sum()),"target_share":float(mask.mean()),"monthly_revenue_exposure":float(target.sum()),"annualized_revenue_exposure":float(target.sum()*12)})
        return pd.DataFrame(rows)

    def retention_scenarios(self,threshold_df,intervention_cost=20.0,success_rates=(0.10,0.20,0.30)):
        if intervention_cost<0: raise ValueError("intervention_cost must be non-negative.")
        rows=[]
        for _,row in threshold_df.iterrows():
            for rate in success_rates:
                if not 0<=rate<=1: raise ValueError("success rates must be between 0 and 1.")
                exposure=float(row["monthly_revenue_exposure"]); customers=int(row["customers_targeted"])
                cost=customers*intervention_cost; retained=exposure*rate
                rows.append({"threshold":float(row["threshold"]),"success_rate":float(rate),"customers_targeted":customers,"monthly_revenue_exposure":exposure,"intervention_cost":cost,"expected_retained_monthly_value":retained,"expected_net_monthly_value":retained-cost,"expected_retained_annual_value":retained*12,"assumptions":"Hypothetical scenario; success rate and intervention cost are assumptions, not observed outcomes."})
        return pd.DataFrame(rows)

    def segment_by_risk_level(self,df,p):
        p=self._validate_inputs(df,p); out=df.copy(); out["ChurnProbability"]=p
        out["RiskSegment"]=np.select([p>=self.risk_threshold,(p>=0.4)&(p<self.risk_threshold)],["High Risk","Medium Risk"],default="Low Risk")
        return out
