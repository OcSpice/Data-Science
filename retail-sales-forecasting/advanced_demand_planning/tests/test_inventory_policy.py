import pandas as pd

from advanced_demand_planning.src.inventory_policy import calculate_inventory_scenario


def test_reorder_point_increases_with_demand_scenario():
    base = pd.DataFrame({
        "item_id": ["A"],
        "store_id": ["S1"],
        "forecast_demand": [10.0],
    })
    low = calculate_inventory_scenario(base, 2.0, demand_multiplier=0.9)
    high = calculate_inventory_scenario(base, 2.0, demand_multiplier=1.1)
    assert high.loc[0, "reorder_point"] > low.loc[0, "reorder_point"]
