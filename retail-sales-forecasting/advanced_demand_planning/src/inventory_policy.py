from __future__ import annotations

import numpy as np
import pandas as pd

Z_VALUES = {0.90: 1.282, 0.95: 1.645, 0.975: 1.960, 0.99: 2.326}


def z_for_service_level(service_level: float) -> float:
    if service_level not in Z_VALUES:
        raise ValueError("Supported service levels: 0.90, 0.95, 0.975, 0.99")
    return Z_VALUES[service_level]


def calculate_inventory_scenario(
    demand_forecast: pd.DataFrame,
    demand_error_std: float,
    lead_time_days: int = 7,
    service_level: float = 0.95,
    demand_multiplier: float = 1.0,
) -> pd.DataFrame:
    """Calculate scenario reorder points.

    This is a planning simulation, not historical inventory reconstruction.
    """
    required = {"item_id", "store_id", "forecast_demand"}
    missing = required - set(demand_forecast.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    out = demand_forecast.copy()
    z = z_for_service_level(service_level)
    out["scenario_demand"] = out["forecast_demand"].clip(lower=0) * demand_multiplier
    out["error_std"] = float(max(demand_error_std, 0))
    out["lead_time_demand"] = out["scenario_demand"] * lead_time_days
    out["safety_stock"] = z * out["error_std"] * np.sqrt(lead_time_days)
    out["reorder_point"] = out["lead_time_demand"] + out["safety_stock"]
    return out


def compare_scenarios(
    base_forecast: pd.DataFrame,
    error_std: float,
    lead_time_days: int = 7,
    service_level: float = 0.95,
) -> pd.DataFrame:
    frames = []
    for name, multiplier in [
        ("Low demand", 0.90),
        ("Base demand", 1.00),
        ("High demand", 1.10),
    ]:
        x = calculate_inventory_scenario(
            base_forecast, error_std, lead_time_days, service_level, multiplier
        )
        x["scenario"] = name
        frames.append(x)
    return pd.concat(frames, ignore_index=True)
