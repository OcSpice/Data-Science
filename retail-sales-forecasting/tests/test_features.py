import pandas as pd
from src.features import prepare_model_features

def test_model_features_exclude_target():
    df = pd.DataFrame({
        "id":["A"]*30,
        "date":pd.date_range("2025-01-01",periods=30),
        "sales":range(30),
        "wday":[1]*30,"month":[1]*30,
        "store_id":["CA_1"]*30,"state_id":["CA"]*30,
        "item_id":["FOODS_1_001"]*30,"dept_id":["FOODS_1"]*30,
        "cat_id":["FOODS"]*30,
    })
    X=prepare_model_features(df)
    assert "sales" not in X.columns
    assert "lag_7" in X.columns
