import pandas as pd

from advanced_demand_planning.src.demand_profile import profile_item_store_demand


def test_profile_builds_series():
    df = pd.DataFrame({
        "item_id": ["A"] * 4,
        "store_id": ["S1"] * 4,
        "date": pd.date_range("2025-01-01", periods=4),
        "sales": [1, 2, 0, 3],
    })
    out = profile_item_store_demand(df)
    assert len(out) == 1
    assert out.loc[0, "total_demand"] == 6
