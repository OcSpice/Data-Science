"""
Data Validator Module

Handles data quality validation, schema checks, and anonymization.
Author: OGHENEOCHUKO EMMANUEL OGIDIAGBA
"""

import hashlib
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple


class DataValidator:
    """
    Validates data quality, enforces schema, and handles anonymization.
    
    Attributes:
        expected_columns (List[str]): List of expected column names.
        numeric_columns (List[str]): Columns that should be numeric.
        categorical_columns (List[str]): Columns that should be categorical.
    """
    
    EXPECTED_COLUMNS = [
        "CustomerID", "Gender", "SeniorCitizen", "Partner", "Dependents",
        "Tenure_Months", "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
        "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn", "TenureQ"
    ]
    
    NUMERIC_COLUMNS = ["SeniorCitizen", "Tenure_Months", "MonthlyCharges", "TotalCharges", "Churn"]
    
    CATEGORICAL_COLUMNS = [
        "Gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
        "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
        "PaperlessBilling", "PaymentMethod", "TenureQ"
    ]
    
    def __init__(self):
        """Initialize the DataValidator."""
        self.validation_errors: List[Dict[str, Any]] = []
        self.validation_warnings: List[Dict[str, Any]] = []
    
    def validate_schema(self, df: pd.DataFrame) -> bool:
        """
        Validate that the DataFrame has all expected columns.
        
        Args:
            df: Input DataFrame to validate.
            
        Returns:
            bool: True if schema is valid, False otherwise.
        """
        missing_cols = set(self.EXPECTED_COLUMNS) - set(df.columns)
        extra_cols = set(df.columns) - set(self.EXPECTED_COLUMNS)
        
        if missing_cols:
            self.validation_errors.append({
                "type": "missing_columns",
                "columns": list(missing_cols)
            })
            return False
        
        if extra_cols:
            self.validation_warnings.append({
                "type": "extra_columns",
                "columns": list(extra_cols)
            })
        
        return len(missing_cols) == 0
    
    def validate_data_types(self, df: pd.DataFrame) -> bool:
        """
        Validate that numeric columns have appropriate data types.
        
        Args:
            df: Input DataFrame to validate.
            
        Returns:
            bool: True if data types are valid, False otherwise.
        """
        is_valid = True
        
        for col in self.NUMERIC_COLUMNS:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    self.validation_warnings.append({
                        "type": "non_numeric_column",
                        "column": col,
                        "current_type": str(df[col].dtype)
                    })
        
        return is_valid
    
    def check_missing_values(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        Check for missing values in the dataset.
        
        Args:
            df: Input DataFrame to check.
            
        Returns:
            Dict[str, int]: Dictionary mapping column names to missing value counts.
        """
        missing_counts = df.isnull().sum().to_dict()
        
        for col, count in missing_counts.items():
            if count > 0:
                self.validation_warnings.append({
                    "type": "missing_values",
                    "column": col,
                    "count": int(count)
                })
        
        return missing_counts
    
    def check_duplicates(self, df: pd.DataFrame, subset: str = "CustomerID") -> int:
        """
        Check for duplicate records.
        
        Args:
            df: Input DataFrame to check.
            subset: Column to check for duplicates.
            
        Returns:
            int: Number of duplicate records.
        """
        duplicate_count = df.duplicated(subset=[subset]).sum()
        
        if duplicate_count > 0:
            self.validation_warnings.append({
                "type": "duplicate_records",
                "column": subset,
                "count": int(duplicate_count)
            })
        
        return int(duplicate_count)
    
    def anonymize_customer_id(self, df: pd.DataFrame, 
                               column: str = "CustomerID") -> pd.DataFrame:
        """
        Hash CustomerID values for privacy compliance.
        
        Args:
            df: Input DataFrame.
            column: Name of the column containing customer IDs.
            
        Returns:
            pd.DataFrame: DataFrame with hashed customer IDs.
        """
        if column not in df.columns:
            raise ValueError(f"Column {column} not found in DataFrame")
        
        df_anonymized = df.copy()
        df_anonymized[column] = df_anonymized[column].apply(
            lambda x: hashlib.sha256(str(x).encode()).hexdigest()[:16]
        )
        
        return df_anonymized
    
    def run_full_validation(self, df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """
        Run complete validation pipeline.
        
        Args:
            df: Input DataFrame to validate.
            
        Returns:
            Tuple[bool, Dict[str, Any]]: Validation status and summary report.
        """
        self.validation_errors = []
        self.validation_warnings = []
        
        schema_valid = self.validate_schema(df)
        self.validate_data_types(df)
        missing_values = self.check_missing_values(df)
        duplicate_count = self.check_duplicates(df)
        
        is_valid = len(self.validation_errors) == 0
        
        report = {
            "is_valid": is_valid,
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "missing_values": missing_values,
            "duplicate_count": duplicate_count,
            "errors": self.validation_errors,
            "warnings": self.validation_warnings
        }
        
        return is_valid, report
