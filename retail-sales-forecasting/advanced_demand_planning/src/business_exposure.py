from __future__ import annotations

import pandas as pd


def summarize_exposure(
    scenario_df: pd.DataFrame,
    inventory_column: str = "inventory_position",
) -> pd.DataFrame:
    """Summarize planning exposure when an assumed inventory position is supplied."""
    required = {
        "item_id",
        "store_id",
        "scenario",
        "reorder_point",
        "scenario_demand",
        inventory_column,
    }
    missing = required - set(scenario_df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    out = scenario_df.copy()
    out["inventory_gap"] = out[inventory_column] - out["reorder_point"]
    out["stockout_risk_flag"] = out["inventory_gap"] < 0
    out["excess_inventory_flag"] = out["inventory_gap"] > out["scenario_demand"]

    return out.groupby("scenario", as_index=False).agg(
        total_inventory=(inventory_column, "sum"),
        total_reorder_requirement=("reorder_point", "sum"),
        stockout_risk_count=("stockout_risk_flag", "sum"),
        excess_inventory_count=("excess_inventory_flag", "sum"),
        total_inventory_gap=("inventory_gap", "sum"),
    )
