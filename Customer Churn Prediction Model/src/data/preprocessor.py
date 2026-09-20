"""
Data Preprocessor Module

Handles data cleaning, transformation, and feature engineering.
Author: OGHENEOCHUKO EMMANUEL OGIDIAGBA
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Optional


class DataPreprocessor:
    """
    Preprocesses customer churn data for modeling.
    
    Handles missing value imputation, feature encoding, and train-test splitting.
    
    Attributes:
        categorical_features (List[str]): List of categorical feature names.
        numeric_features (List[str]): List of numeric feature names.
        target_column (str): Name of the target column.
    """
    
    CATEGORICAL_FEATURES = [
        "Gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
        "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
        "PaperlessBilling", "PaymentMethod", "TenureQ"
    ]
    
    NUMERIC_FEATURES = ["SeniorCitizen", "Tenure_Months", "MonthlyCharges", "TotalCharges"]
    
    TARGET_COLUMN = "Churn"
    
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        """
        Initialize the DataPreprocessor.
        
        Args:
            test_size: Proportion of data to use for testing.
            random_state: Random seed for reproducibility.
        """
        self.test_size = test_size
        self.random_state = random_state
        self.feature_columns: List[str] = []
        self.label_encoders: dict = {}
        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the dataset by handling missing values and data type issues.
        
        Args:
            df: Input DataFrame.
            
        Returns:
            pd.DataFrame: Cleaned DataFrame.
        """
        df_clean = df.copy()
        
        if "TotalCharges" in df_clean.columns:
            df_clean["TotalCharges"] = pd.to_numeric(
                df_clean["TotalCharges"], errors="coerce"
            )
            median_total_charges = df_clean["TotalCharges"].median()
            df_clean["TotalCharges"] = df_clean["TotalCharges"].fillna(
                median_total_charges
            )
        
        if "Tenure_Months" in df_clean.columns:
            median_tenure = df_clean["Tenure_Months"].median()
            df_clean["Tenure_Months"] = df_clean["Tenure_Months"].fillna(median_tenure)
        
        for col in self.CATEGORICAL_FEATURES:
            if col in df_clean.columns:
                mode_value = df_clean[col].mode()[0] if not df_clean[col].mode().empty else "Unknown"
                df_clean[col] = df_clean[col].fillna(mode_value)
        
        return df_clean
    
    def encode_categorical_features(self, df: pd.DataFrame, 
                                     fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical features using label encoding.
        
        Args:
            df: Input DataFrame.
            fit: Whether to fit new encoders or use existing ones.
            
        Returns:
            pd.DataFrame: DataFrame with encoded categorical features.
        """
        df_encoded = df.copy()
        
        if fit:
            self.label_encoders = {}
        
        for col in self.CATEGORICAL_FEATURES:
            if col in df_encoded.columns:
                if fit:
                    unique_values = df_encoded[col].unique()
                    self.label_encoders[col] = {
                        val: idx for idx, val in enumerate(unique_values)
                    }
                
                encoder_map = self.label_encoders[col]
                df_encoded[col] = df_encoded[col].map(
                    lambda x: encoder_map.get(x, -1)
                )
        
        return df_encoded
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for modeling by selecting and ordering columns.
        
        Args:
            df: Input DataFrame.
            
        Returns:
            pd.DataFrame: DataFrame with prepared features.
        """
        self.feature_columns = self.NUMERIC_FEATURES + self.CATEGORICAL_FEATURES
        
        available_features = [
            col for col in self.feature_columns if col in df.columns
        ]
        
        return df[available_features].copy()
    
    def split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.Series, pd.Series
    ]:
        """
        Split data into training and testing sets.
        
        Args:
            X: Feature matrix.
            y: Target vector.
            
        Returns:
            Tuple containing X_train, X_test, y_train, y_test.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=y
        )
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def preprocess_full(self, df: pd.DataFrame) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.Series, pd.Series
    ]:
        """
        Run complete preprocessing pipeline.
        
        Args:
            df: Input DataFrame.
            
        Returns:
            Tuple containing X_train, X_test, y_train, y_test.
        """
        df_clean = self.clean_data(df)
        df_encoded = self.encode_categorical_features(df_clean, fit=True)
        X = self.prepare_features(df_encoded)
        y = df_encoded[self.TARGET_COLUMN]
        
        return self.split_data(X, y)
