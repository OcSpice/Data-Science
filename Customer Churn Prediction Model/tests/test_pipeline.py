"""Tests for the churn prediction and retention economics pipeline."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

BASE_DIR=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(BASE_DIR/"src"))

from data.preprocessor import DataPreprocessor
from models.classifier import ChurnClassifier
from business_impact.revenue_calculator import RevenueCalculator

@pytest.fixture
def sample_data():
    rng=np.random.default_rng(42); n=240
    return pd.DataFrame({
        "CustomerID":[f"C{i:05d}" for i in range(n)],
        "Gender":rng.choice(["Male","Female"],n),"SeniorCitizen":rng.integers(0,2,n),
        "Partner":rng.choice(["Yes","No"],n),"Dependents":rng.choice(["Yes","No"],n),
        "Tenure_Months":rng.integers(1,72,n),"PhoneService":rng.choice(["Yes","No"],n),
        "MultipleLines":rng.choice(["Yes","No","No phone service"],n),
        "InternetService":rng.choice(["DSL","Fiber optic","No"],n),
        "OnlineSecurity":rng.choice(["Yes","No","No internet service"],n),
        "OnlineBackup":rng.choice(["Yes","No","No internet service"],n),
        "DeviceProtection":rng.choice(["Yes","No","No internet service"],n),
        "TechSupport":rng.choice(["Yes","No","No internet service"],n),
        "StreamingTV":rng.choice(["Yes","No","No internet service"],n),
        "StreamingMovies":rng.choice(["Yes","No","No internet service"],n),
        "Contract":rng.choice(["Month-to-month","One year","Two year"],n),
        "PaperlessBilling":rng.choice(["Yes","No"],n),
        "PaymentMethod":rng.choice(["Electronic check","Mailed check","Bank transfer","Credit card"],n),
        "MonthlyCharges":rng.uniform(20,120,n),"TotalCharges":rng.uniform(100,8000,n),
        "Churn":rng.choice([0,1],n,p=[0.73,0.27]),
        "TenureQ":rng.choice(["Q1 (0-6mo)","Q2 (7-12mo)","Q3 (13-24mo)","Q4 (25-48mo)","Q5 (49-72mo)"],n)
    })

def test_preprocessing_is_leakage_safe(sample_data):
    prep=DataPreprocessor()
    X_train,X_test,y_train,y_test=prep.split_raw_data(sample_data)
    prep.fit_transformers(X_train)
    assert len(X_train)+len(X_test)==len(sample_data)
    assert set(X_train.index).isdisjoint(set(X_test.index))
    assert prep.get_feature_names("tree")

def test_model_comparison_and_metrics(sample_data):
    prep=DataPreprocessor(); X_train,X_test,y_train,y_test=prep.split_raw_data(sample_data)
    clf=ChurnClassifier(prep); clf.fit(X_train,y_train,tune_params=False)
    metrics=clf.evaluate(X_test,y_test)
    assert set(metrics)=={"majority_baseline","logistic_regression","random_forest"}
    for values in metrics.values():
        assert 0<=values["roc_auc"]<=1
        assert 0<=values["pr_auc"]<=1

def test_threshold_and_scenario_analysis(sample_data):
    calc=RevenueCalculator(0.7); probabilities=np.linspace(0,1,len(sample_data))
    thresholds=calc.threshold_analysis(sample_data,probabilities)
    scenarios=calc.retention_scenarios(thresholds,intervention_cost=20,success_rates=(0.1,0.2,0.3))
    assert len(thresholds)==6
    assert len(scenarios)==18
    assert (scenarios["intervention_cost"]>=0).all()

def test_invalid_scenario_inputs(sample_data):
    calc=RevenueCalculator()
    with pytest.raises(ValueError):
        calc.retention_scenarios(pd.DataFrame([{"threshold":0.5,"customers_targeted":1,"monthly_revenue_exposure":10}]),intervention_cost=-1)
