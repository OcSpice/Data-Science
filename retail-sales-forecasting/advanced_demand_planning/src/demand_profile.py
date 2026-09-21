from __future__ import annotations

import numpy as np
import pandas as pd


def profile_item_store_demand(df: pd.DataFrame) -> pd.DataFrame:
    """Build history-only demand profiles at item-store level."""
    required = {"item_id", "store_id", "date", "sales"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    work = df.copy()
    work["date"] = pd.to_datetime(work["date"])

    g = work.groupby(["item_id", "store_id"], as_index=False)
    out = g["sales"].agg(
        mean_daily_demand="mean",
        std_daily_demand="std",
        total_demand="sum",
        zero_demand_rate=lambda s: (s == 0).mean(),
    )
    out["std_daily_demand"] = out["std_daily_demand"].fillna(0)
    out["cv"] = np.where(
        out["mean_daily_demand"] > 0,
        out["std_daily_demand"] / out["mean_daily_demand"],
        np.nan,
    )

    work = work.sort_values("date")
    recent = work.groupby(["item_id", "store_id"]).tail(28).groupby(
        ["item_id", "store_id"], as_index=False
    )["sales"].mean().rename(columns={"sales": "recent_28d_mean"})
    early = work.groupby(["item_id", "store_id"]).head(28).groupby(
        ["item_id", "store_id"], as_index=False
    )["sales"].mean().rename(columns={"sales": "early_28d_mean"})

    out = out.merge(recent, on=["item_id", "store_id"], how="left")
    out = out.merge(early, on=["item_id", "store_id"], how="left")
    out["trend_ratio"] = np.where(
        out["early_28d_mean"] > 0,
        out["recent_28d_mean"] / out["early_28d_mean"],
        np.nan,
    )
    out["demand_segment"] = classify_segments(out)
    return out


def classify_segments(profile: pd.DataFrame) -> pd.Series:
    # Rank by position rather than pd.cut on raw quantiles so tied demand
    # values cannot create duplicate bin edges on small or discrete samples.
    ranks = profile["mean_daily_demand"].rank(method="first", pct=True)
    volume = pd.cut(
        ranks,
        bins=[-np.inf, 1 / 3, 2 / 3, np.inf],
        labels=["Low volume", "Medium volume", "High volume"],
    )

    intermittent = profile["zero_demand_rate"].ge(0.10)
    volatile = profile["cv"].ge(1.0)
    growing = profile["trend_ratio"].gt(1.15)
    declining = profile["trend_ratio"].lt(0.85)

    segment = pd.Series("Stable", index=profile.index, dtype="object")
    segment.loc[volatile] = "Volatile"
    segment.loc[intermittent] = "Intermittent"
    segment.loc[growing & ~intermittent] = "Growing"
    segment.loc[declining & ~intermittent] = "Declining"

    return volume.astype(str) + " / " + segment
