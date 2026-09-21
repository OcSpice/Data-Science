from src.data import load_training,load_validation,load_test

def test_chronological_split():
    train=load_training(); val=load_validation(); test=load_test()
    assert train["date"].max()<val["date"].min()
    assert val["date"].max()<test["date"].min()
    assert "sales" in train.columns and "sales" in val.columns
    assert "sales" not in test.columns
