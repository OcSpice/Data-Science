import pandas as pd

from advanced_demand_planning.src.business_exposure import summarize_exposure


def test_exposure_flags_stockout_and_excess():
    df = pd.DataFrame({
        "item_id": ["a", "b"],
        "store_id": ["s", "s"],
        "scenario": ["Base demand", "Base demand"],
        "reorder_point": [100.0, 20.0],
        "scenario_demand": [10.0, 10.0],
        "inventory_position": [80.0, 40.0],
    })
    out = summarize_exposure(df)
    assert out.loc[0, "stockout_risk_count"] == 1
    assert out.loc[0, "excess_inventory_count"] == 1
