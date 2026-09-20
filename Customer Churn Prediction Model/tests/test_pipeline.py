"""
Unit Tests for Customer Churn Prediction Pipeline

Tests data transformation, model performance thresholds, and revenue calculation.
Author: OGHENEOCHUKO EMMANUEL OGIDIAGBA
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

BASE_DIR = Path(__file__).parent.parent
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from src.data.validator import DataValidator
from src.data.preprocessor import DataPreprocessor
from src.models.classifier import ChurnClassifier
from src.business_impact.revenue_calculator import RevenueCalculator


@pytest.fixture
def sample_data():
    """Create sample test data."""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        "CustomerID": [f"CUST-{i:06d}" for i in range(n_samples)],
        "Gender": np.random.choice(["Male", "Female"], n_samples),
        "SeniorCitizen": np.random.choice([0, 1], n_samples),
        "Partner": np.random.choice(["Yes", "No"], n_samples),
        "Dependents": np.random.choice(["Yes", "No"], n_samples),
        "Tenure_Months": np.random.randint(1, 72, n_samples),
        "PhoneService": np.random.choice(["Yes", "No"], n_samples),
        "MultipleLines": np.random.choice(["Yes", "No", "No phone service"], n_samples),
        "InternetService": np.random.choice(["DSL", "Fiber optic", "No"], n_samples),
        "OnlineSecurity": np.random.choice(["Yes", "No", "No internet service"], n_samples),
        "OnlineBackup": np.random.choice(["Yes", "No", "No internet service"], n_samples),
        "DeviceProtection": np.random.choice(["Yes", "No", "No internet service"], n_samples),
        "TechSupport": np.random.choice(["Yes", "No", "No internet service"], n_samples),
        "StreamingTV": np.random.choice(["Yes", "No", "No internet service"], n_samples),
        "StreamingMovies": np.random.choice(["Yes", "No", "No internet service"], n_samples),
        "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], n_samples),
        "PaperlessBilling": np.random.choice(["Yes", "No"], n_samples),
        "PaymentMethod": np.random.choice([
            "Electronic check", "Mailed check", "Bank transfer", "Credit card"
        ], n_samples),
        "MonthlyCharges": np.random.uniform(20, 120, n_samples),
        "TotalCharges": np.random.uniform(100, 8000, n_samples),
        "Churn": np.random.choice([0, 1], n_samples, p=[0.73, 0.27]),
        "TenureQ": np.random.choice(["Q1 (0-6mo)", "Q2 (7-12mo)", "Q3 (13-24mo)", 
                                      "Q4 (25-48mo)", "Q5 (49-72mo)"], n_samples)
    }
    
    return pd.DataFrame(data)


class TestDataValidator:
    """Tests for DataValidator class."""
    
    def test_schema_validation_passes(self, sample_data):
        """Test that valid schema passes validation."""
        validator = DataValidator()
        is_valid, report = validator.run_full_validation(sample_data)
        
        assert is_valid is True
        assert len(report["errors"]) == 0
    
    def test_missing_columns_detected(self, sample_data):
        """Test that missing columns are detected."""
        incomplete_data = sample_data.drop(columns=["Churn"])
        validator = DataValidator()
        
        is_valid = validator.validate_schema(incomplete_data)
        
        assert is_valid is False
        assert len(validator.validation_errors) > 0
    
    def test_anonymization_hashes_customer_id(self, sample_data):
        """Test that CustomerID is properly hashed."""
        validator = DataValidator()
        original_id = sample_data["CustomerID"].iloc[0]
        
        anonymized = validator.anonymize_customer_id(sample_data)
        hashed_id = anonymized["CustomerID"].iloc[0]
        
        assert hashed_id != original_id
        assert len(hashed_id) == 16
        assert hashed_id.isalnum()
    
    def test_missing_values_detected(self, sample_data):
        """Test that missing values are detected."""
        data_with_missing = sample_data.copy()
        data_with_missing.loc[0, "Tenure_Months"] = np.nan
        
        validator = DataValidator()
        missing_counts = validator.check_missing_values(data_with_missing)
        
        assert missing_counts["Tenure_Months"] == 1


class TestDataPreprocessor:
    """Tests for DataPreprocessor class."""
    
    def test_clean_data_handles_total_charges(self, sample_data):
        """Test cleaning handles TotalCharges conversion."""
        preprocessor = DataPreprocessor()
        sample_data.loc[0, "TotalCharges"] = " "
        
        cleaned = preprocessor.clean_data(sample_data)
        
        assert pd.api.types.is_numeric_dtype(cleaned["TotalCharges"])
        assert not cleaned["TotalCharges"].isna().any()
    
    def test_encode_categorical_features(self, sample_data):
        """Test categorical encoding works correctly."""
        preprocessor = DataPreprocessor()
        encoded = preprocessor.encode_categorical_features(sample_data, fit=True)
        
        assert encoded["Contract"].dtype in [np.int64, np.int32]
        assert encoded["Gender"].dtype in [np.int64, np.int32]
    
    def test_split_data_preserves_stratification(self, sample_data):
        """Test train-test split preserves class distribution."""
        preprocessor = DataPreprocessor(test_size=0.3, random_state=42)
        
        X = sample_data[["Tenure_Months", "MonthlyCharges"]]
        y = sample_data["Churn"]
        
        X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
        
        original_ratio = y.mean()
        train_ratio = y_train.mean()
        test_ratio = y_test.mean()
        
        assert abs(original_ratio - train_ratio) < 0.1
        assert abs(original_ratio - test_ratio) < 0.1


class TestChurnClassifier:
    """Tests for ChurnClassifier class."""
    
    def test_model_training(self, sample_data):
        """Test model training completes successfully."""
        preprocessor = DataPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.preprocess_full(sample_data)
        
        classifier = ChurnClassifier(random_state=42)
        classifier.fit(X_train, y_train, use_smote=False, tune_params=False)
        
        assert classifier._is_trained is True
        assert classifier.model is not None
    
    def test_prediction_output_shape(self, sample_data):
        """Test predictions have correct shape."""
        preprocessor = DataPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.preprocess_full(sample_data)
        
        classifier = ChurnClassifier(random_state=42)
        classifier.fit(X_train, y_train, use_smote=False, tune_params=False)
        
        predictions = classifier.predict(X_test)
        probabilities = classifier.predict_proba(X_test)
        
        assert len(predictions) == len(X_test)
        assert probabilities.shape[1] == 2
    
    def test_roc_auc_threshold(self, sample_data):
        """Test ROC-AUC meets minimum threshold of 0.75."""
        preprocessor = DataPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.preprocess_full(sample_data)
        
        classifier = ChurnClassifier(random_state=42)
        classifier.fit(X_train, y_train, use_smote=True, tune_params=True)
        
        metrics = classifier.evaluate(X_test, y_test)
        
        assert metrics["roc_auc"] >= 0.75, f"ROC-AUC {metrics['roc_auc']} below 0.75 threshold"
    
    def test_feature_importance_computed(self, sample_data):
        """Test feature importances are computed after training."""
        preprocessor = DataPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.preprocess_full(sample_data)
        
        classifier = ChurnClassifier(random_state=42)
        classifier.fit(X_train, y_train, use_smote=False, tune_params=False)
        
        assert len(classifier.feature_importances) > 0
        assert all(v >= 0 for v in classifier.feature_importances.values())


class TestRevenueCalculator:
    """Tests for RevenueCalculator class."""
    
    def test_at_risk_revenue_calculation(self, sample_data):
        """Test at-risk revenue is calculated correctly."""
        calculator = RevenueCalculator(risk_threshold=0.7)
        
        churn_probs = np.random.uniform(0, 1, len(sample_data))
        
        summary = calculator.calculate_at_risk_revenue(sample_data, churn_probs)
        
        assert "at_risk_revenue_monthly" in summary
        assert "high_risk_customers" in summary
        assert summary["at_risk_revenue_monthly"] >= 0
    
    def test_high_risk_segmentation(self, sample_data):
        """Test customers are correctly segmented by risk."""
        calculator = RevenueCalculator(risk_threshold=0.7)
        
        churn_probs = np.array([0.8, 0.5, 0.3, 0.9, 0.2])
        test_data = sample_data.head(5).copy()
        
        segmented = calculator.segment_by_risk_level(test_data, churn_probs)
        
        assert segmented.loc[0, "RiskSegment"] == "High Risk"
        assert segmented.loc[1, "RiskSegment"] == "Medium Risk"
        assert segmented.loc[2, "RiskSegment"] == "Low Risk"
    
    def test_author_metadata_persistence(self, sample_data):
        """Test author metadata is included in reports."""
        calculator = RevenueCalculator(risk_threshold=0.7)
        
        churn_probs = np.random.uniform(0, 1, len(sample_data))
        summary = calculator.calculate_at_risk_revenue(sample_data, churn_probs)
        
        assert summary["author"] == "OGHENEOCHUKO EMMANUEL OGIDIAGBA"
        assert summary["project"] == "Customer Churn Prediction Model"
    
    def test_revenue_aggregation_logic(self):
        """Test revenue aggregation logic with known values."""
        calculator = RevenueCalculator(risk_threshold=0.5)
        
        test_data = pd.DataFrame({
            "CustomerID": ["C1", "C2", "C3", "C4"],
            "MonthlyCharges": [100, 50, 80, 30]
        })
        churn_probs = np.array([0.8, 0.3, 0.6, 0.4])
        
        summary = calculator.calculate_at_risk_revenue(test_data, churn_probs)
        
        expected_revenue = 100 + 80
        assert summary["at_risk_revenue_monthly"] == expected_revenue
        assert summary["high_risk_customers"] == 2


class TestIntegration:
    """Integration tests for the complete pipeline."""
    
    def test_end_to_end_preprocessing_and_modeling(self, sample_data):
        """Test complete preprocessing and modeling flow."""
        preprocessor = DataPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.preprocess_full(sample_data)
        
        classifier = ChurnClassifier(random_state=42)
        classifier.fit(X_train, y_train, use_smote=True, tune_params=False)
        metrics = classifier.evaluate(X_test, y_test)
        
        assert metrics["accuracy"] > 0.5
        assert "f1_score" in metrics
    
    def test_validator_preprocessor_integration(self, sample_data):
        """Test validator and preprocessor work together."""
        validator = DataValidator()
        preprocessor = DataPreprocessor()
        
        is_valid, _ = validator.run_full_validation(sample_data)
        assert is_valid is True
        
        X_train, X_test, y_train, y_test = preprocessor.preprocess_full(sample_data)
        
        assert len(X_train) > 0
        assert len(y_test) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
