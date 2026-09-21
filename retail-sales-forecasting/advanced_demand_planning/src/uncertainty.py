from __future__ import annotations

import pandas as pd


def summarize_uncertainty(
    forecast_df: pd.DataFrame,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Summarize historical forecast error as an uncertainty proxy."""
    required = {"actual", "prediction"}
    missing = required - set(forecast_df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    group_cols = group_cols or []
    work = forecast_df.copy()
    work["error"] = work["actual"] - work["prediction"]
    work["abs_error"] = work["error"].abs()

    if not group_cols:
        return pd.DataFrame([{
            "mae": work["abs_error"].mean(),
            "error_std": work["error"].std(ddof=1),
            "error_p90_abs": work["abs_error"].quantile(0.90),
            "error_p95_abs": work["abs_error"].quantile(0.95),
            "n_obs": len(work),
        }])

    return work.groupby(group_cols, as_index=False).agg(
        mae=("abs_error", "mean"),
        error_std=("error", "std"),
        error_p90_abs=("abs_error", lambda s: s.quantile(0.90)),
        error_p95_abs=("abs_error", lambda s: s.quantile(0.95)),
        n_obs=("error", "size"),
    )
