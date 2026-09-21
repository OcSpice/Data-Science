from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

HERE = Path(__file__).resolve()
PROJECT = HERE.parents[2]
sys.path.insert(0, str(PROJECT))

from src.data import load_training
from advanced_demand_planning.src.demand_profile import profile_item_store_demand
from advanced_demand_planning.src.uncertainty import summarize_uncertainty
from advanced_demand_planning.src.inventory_policy import compare_scenarios
from advanced_demand_planning.src.business_exposure import summarize_exposure


def build_seasonal_error_profile(history: pd.DataFrame) -> pd.DataFrame:
    """Estimate item-store forecast error using a lag-7 naive benchmark.

    This deliberately uses a simple, transparent benchmark so the planning
    layer does not pretend that uncertainty is probabilistically calibrated.
    """
    work = history.copy()
    work["date"] = pd.to_datetime(work["date"])
    work = work.sort_values(["item_id", "store_id", "date"])
    work["prediction"] = work.groupby(["item_id", "store_id"])["sales"].shift(7)
    work = work.dropna(subset=["prediction"]).copy()
    work["actual"] = work["sales"]

    return summarize_uncertainty(
        work,
        group_cols=["item_id", "store_id"],
    )


def build_planning_forecast(profile: pd.DataFrame) -> pd.DataFrame:
    """Use recent 28-day demand as the transparent planning baseline."""
    return profile[
        ["item_id", "store_id", "recent_28d_mean"]
    ].rename(columns={"recent_28d_mean": "forecast_demand"}).copy()


def main() -> None:
    reports = HERE.parent.parent / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    history = load_training()
    profile = profile_item_store_demand(history)
    profile.to_csv(reports / "demand_segments.csv", index=False)

    segment_summary = (
        profile.groupby("demand_segment", as_index=False)
        .agg(
            series=("item_id", "size"),
            mean_daily_demand=("mean_daily_demand", "mean"),
            avg_zero_demand_rate=("zero_demand_rate", "mean"),
            avg_cv=("cv", "mean"),
        )
        .sort_values("series", ascending=False)
    )
    segment_summary.to_csv(reports / "demand_segment_summary.csv", index=False)

    uncertainty = build_seasonal_error_profile(history)
    uncertainty.to_csv(reports / "uncertainty_summary.csv", index=False)

    planning = build_planning_forecast(profile)
    planning = planning.merge(
        uncertainty[["item_id", "store_id", "error_std", "mae", "error_p95_abs"]],
        on=["item_id", "store_id"],
        how="left",
    )
    planning["error_std"] = planning["error_std"].fillna(0.0)

    # Base inventory position is an explicit simulation assumption:
    # seven days of the recent planning baseline.
    planning["inventory_position"] = planning["forecast_demand"] * 7

    scenarios = []
    for _, row in planning.iterrows():
        base = pd.DataFrame([{
            "item_id": row["item_id"],
            "store_id": row["store_id"],
            "forecast_demand": row["forecast_demand"],
        }])
        x = compare_scenarios(
            base_forecast=base,
            error_std=float(row["error_std"]),
            lead_time_days=7,
            service_level=0.95,
        )
        x["inventory_position"] = row["inventory_position"]
        scenarios.append(x)

    scenario_df = pd.concat(scenarios, ignore_index=True)
    scenario_df.to_csv(reports / "inventory_scenarios.csv", index=False)

    exposure = summarize_exposure(scenario_df)
    exposure.to_csv(reports / "business_exposure.csv", index=False)

    print("\nDemand segments")
    print(segment_summary.to_string(index=False))
    print("\nScenario exposure")
    print(exposure.to_string(index=False))


if __name__ == "__main__":
    main()
