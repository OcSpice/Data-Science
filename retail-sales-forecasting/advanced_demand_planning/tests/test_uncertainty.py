import pandas as pd

from advanced_demand_planning.src.uncertainty import summarize_uncertainty


def test_uncertainty_summary_calculates_error_metrics():
    df = pd.DataFrame({
        "item_id": ["a", "a", "a"],
        "store_id": ["s", "s", "s"],
        "actual": [10, 12, 8],
        "prediction": [9, 10, 10],
    })
    out = summarize_uncertainty(df, group_cols=["item_id", "store_id"])
    assert len(out) == 1
    assert out.loc[0, "mae"] == 5 / 3
    assert out.loc[0, "n_obs"] == 3
