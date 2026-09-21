from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd

HERE = Path(__file__).resolve()
PROJECT = HERE.parents[2]
sys.path.insert(0, str(PROJECT))

from src.data import load_training
from advanced_demand_planning.src.demand_profile import profile_item_store_demand


def main() -> None:
    reports = HERE.parent.parent / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    history = load_training()
    profile = profile_item_store_demand(history)
    profile.to_csv(reports / "demand_segments.csv", index=False)

    summary = (
        profile.groupby("demand_segment", as_index=False)
        .agg(
            series=("item_id", "size"),
            mean_daily_demand=("mean_daily_demand", "mean"),
            avg_zero_demand_rate=("zero_demand_rate", "mean"),
            avg_cv=("cv", "mean"),
        )
        .sort_values("series", ascending=False)
    )
    summary.to_csv(reports / "demand_segment_summary.csv", index=False)

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
