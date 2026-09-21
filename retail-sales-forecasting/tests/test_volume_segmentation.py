import pandas as pd
from src.volume_segmentation import classify_series

def test_volume_segmentation_is_history_based():
    df=pd.DataFrame({
        "id":["low"]*10+["high"]*10,
        "sales":[1]*10+[10]*10,
    })
    result=classify_series(df)
    assert set(result["volume_segment"]) == {"Low volume","High volume"}
