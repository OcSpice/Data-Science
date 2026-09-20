"""
Revenue Calculator Module

Quantifies at-risk revenue based on churn predictions.
Author: OGHENEOCHUKO EMMANUEL OGIDIAGBA
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, Optional


class RevenueCalculator:
    """
    Calculates at-risk revenue from customers predicted to churn.
    
    Identifies high-risk customers and aggregates their monthly charges
    to quantify potential revenue loss.
    
    Attributes:
        risk_threshold: Probability threshold for classifying high-risk customers.
        monthly_charges_column: Name of the column containing monthly charges.
    """
    
    AUTHOR = "OGHENEOCHUKO EMMANUEL OGIDIAGBA"
    
    def __init__(self, risk_threshold: float = 0.7,
                 monthly_charges_column: str = "MonthlyCharges"):
        """
        Initialize the RevenueCalculator.
        
        Args:
            risk_threshold: Probability threshold for high-risk classification.
            monthly_charges_column: Column name for monthly charges.
        """
        self.risk_threshold = risk_threshold
        self.monthly_charges_column = monthly_charges_column
        self.at_risk_customers: Optional[pd.DataFrame] = None
        self.total_at_risk_revenue: float = 0.0
        self.summary_report: Dict[str, Any] = {}
    
    def calculate_at_risk_revenue(self, df: pd.DataFrame,
                                   churn_probabilities: np.ndarray) -> Dict[str, Any]:
        """
        Calculate total at-risk revenue from high-churn-probability customers.
        
        Args:
            df: DataFrame with customer data including MonthlyCharges.
            churn_probabilities: Array of churn probabilities from model.
            
        Returns:
            Dict[str, Any]: Summary of at-risk revenue analysis.
        """
        df_analysis = df.copy()
        df_analysis["_churn_probability"] = churn_probabilities
        
        high_risk_mask = df_analysis["_churn_probability"] >= self.risk_threshold
        
        self.at_risk_customers = df_analysis[high_risk_mask].copy()
        
        if self.monthly_charges_column not in self.at_risk_customers.columns:
            raise ValueError(
                f"Column {self.monthly_charges_column} not found in DataFrame"
            )
        
        self.total_at_risk_revenue = float(
            self.at_risk_customers[self.monthly_charges_column].sum()
        )
        
        high_risk_count = len(self.at_risk_customers)
        total_count = len(df_analysis)
        high_risk_percentage = (high_risk_count / total_count * 100) if total_count > 0 else 0
        
        avg_monthly_charge_at_risk = float(
            self.at_risk_customers[self.monthly_charges_column].mean()
        ) if high_risk_count > 0 else 0.0
        
        projected_annual_loss = self.total_at_risk_revenue * 12
        
        self.summary_report = {
            "author": self.AUTHOR,
            "project": "Customer Churn Prediction Model",
            "analysis_type": "At-Risk Revenue Quantification",
            "risk_threshold": self.risk_threshold,
            "total_customers_analyzed": int(total_count),
            "high_risk_customers": int(high_risk_count),
            "high_risk_percentage": round(high_risk_percentage, 2),
            "monthly_charges_column": self.monthly_charges_column,
            "at_risk_revenue_monthly": round(self.total_at_risk_revenue, 2),
            "at_risk_revenue_formatted": f"${self.total_at_risk_revenue:,.2f}",
            "projected_annual_loss": round(projected_annual_loss, 2),
            "projected_annual_loss_formatted": f"${projected_annual_loss:,.2f}",
            "average_monthly_charge_at_risk": round(avg_monthly_charge_at_risk, 2),
            "key_finding": f"${self.total_at_risk_revenue:,.0f} in at-risk revenue identified",
            "recommendations": [
                "Implement targeted retention campaigns for high-risk customers",
                "Offer contract incentives to month-to-month customers",
                "Proactive outreach for customers with fiber optic service",
                "Enhance tech support offerings to reduce churn drivers"
            ]
        }
        
        return self.summary_report
    
    def get_high_risk_customers(self, 
                                 include_details: bool = True) -> pd.DataFrame:
        """
        Get DataFrame of high-risk customers.
        
        Args:
            include_details: Whether to include all customer details.
            
        Returns:
            pd.DataFrame: High-risk customer data.
        """
        if self.at_risk_customers is None:
            raise ValueError("No at-risk customers calculated yet")
        
        if include_details:
            return self.at_risk_customers
        
        return self.at_risk_customers[[
            "_churn_probability", self.monthly_charges_column
        ]]
    
    def segment_by_risk_level(self, df: pd.DataFrame,
                               churn_probabilities: np.ndarray) -> pd.DataFrame:
        """
        Segment customers by risk level.
        
        Args:
            df: DataFrame with customer data.
            churn_probabilities: Array of churn probabilities.
            
        Returns:
            pd.DataFrame: DataFrame with risk segments.
        """
        df_segmented = df.copy()
        df_segmented["_churn_probability"] = churn_probabilities
        
        conditions = [
            df_segmented["_churn_probability"] >= 0.7,
            (df_segmented["_churn_probability"] >= 0.4) & 
            (df_segmented["_churn_probability"] < 0.7),
            df_segmented["_churn_probability"] < 0.4
        ]
        
        choices = ["High Risk", "Medium Risk", "Low Risk"]
        
        df_segmented["RiskSegment"] = np.select(conditions, choices, default="Unknown")
        
        return df_segmented
    
    def save_report(self, output_path: str) -> None:
        """
        Save the revenue analysis report to JSON.
        
        Args:
            output_path: Path to save the report.
        """
        if not self.summary_report:
            raise ValueError("No report generated yet")
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(self.summary_report, f, indent=2)
    
    def generate_executive_summary(self) -> str:
        """
        Generate an executive summary of the revenue impact.
        
        Returns:
            str: Formatted executive summary text.
        """
        if not self.summary_report:
            raise ValueError("No analysis performed yet")
        
        summary = f"""
================================================================================
                    AT-RISK REVENUE ANALYSIS - EXECUTIVE SUMMARY
================================================================================
                    Author: {self.AUTHOR}
                    Project: Customer Churn Prediction Model
================================================================================

KEY METRICS:
------------
* Total Customers Analyzed: {self.summary_report['total_customers_analyzed']:,}
* High-Risk Customers Identified: {self.summary_report['high_risk_customers']:,} 
  ({self.summary_report['high_risk_percentage']}% of base)
* Risk Threshold Applied: {self.summary_report['risk_threshold'] * 100}% churn probability

FINANCIAL IMPACT:
-----------------
* Monthly At-Risk Revenue: {self.summary_report['at_risk_revenue_formatted']}
* Projected Annual Loss: {self.summary_report['projected_annual_loss_formatted']}
* Avg. Monthly Charge (At-Risk): ${self.summary_report['average_monthly_charge_at_risk']:,.2f}

PRIMARY FINDING:
----------------
{self.summary_report['key_finding']} in monthly revenue is at risk from customer churn.

RECOMMENDED ACTIONS:
--------------------
"""
        for i, rec in enumerate(self.summary_report.get("recommendations", []), 1):
            summary += f"{i}. {rec}\n"
        
        summary += """
================================================================================
                         END OF EXECUTIVE SUMMARY
================================================================================
"""
        return summary
