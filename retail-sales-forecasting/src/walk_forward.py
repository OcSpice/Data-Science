"""Walk-forward evaluation utilities for daily retail forecasting.

The evaluator keeps each validation window strictly chronological:
fit -> forecast the next horizon -> score -> move the cutoff forward.
It is designed for lightweight baselines so model selection does not depend
on a single arbitrary validation window.
"""

from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

from .metrics import evaluate


@dataclass(frozen=True)
class Window:
    train_end: str
    forecast_start: str
    forecast_end: str


def make_windows(
    first_train_end: str,
    first_forecast_start: str,
    n_windows: int = 3,
    horizon_days: int = 28,
    step_days: int = 28,
) -> list[Window]:
    """Create consecutive, non-overlapping forecast windows."""
    train_end = pd.Timestamp(first_train_end)
    forecast_start = pd.Timestamp(first_forecast_start)
    windows: list[Window] = []

    for _ in range(n_windows):
        forecast_end = forecast_start + pd.Timedelta(days=horizon_days - 1)
        windows.append(
            Window(
                train_end=train_end.strftime("%Y-%m-%d"),
                forecast_start=forecast_start.strftime("%Y-%m-%d"),
                forecast_end=forecast_end.strftime("%Y-%m-%d"),
            )
        )
        train_end = train_end + pd.Timedelta(days=step_days)
        forecast_start = forecast_start + pd.Timedelta(days=step_days)

    return windows


def seasonal_naive_forecast(
    train: pd.DataFrame,
    forecast: pd.DataFrame,
    season_length: int = 7,
) -> pd.Series:
    """Forecast each item-store series using the last seasonal observation.

    For a 7-day seasonal naive model, each forecast date uses sales from the
    same item-store series seven days earlier. Missing history falls back to
    the series mean from the available training data.
    """
    history = train.copy()
    future = forecast.copy()
    history["date"] = pd.to_datetime(history["date"])
    future["date"] = pd.to_datetime(future["date"])

    lookup = history.set_index(["id", "date"])["sales"]
    keys = pd.MultiIndex.from_arrays(
        [
            future["id"].to_numpy(),
            (future["date"] - pd.Timedelta(days=season_length)).to_numpy(),
        ],
        names=["id", "date"],
    )
    pred = lookup.reindex(keys).to_numpy()

    fallback = history.groupby("id")["sales"].mean()
    fallback_values = future["id"].map(fallback).to_numpy()
    pred = pd.Series(pred).fillna(pd.Series(fallback_values)).to_numpy()
    return pd.Series(pred, index=future.index, name="prediction")


def evaluate_walk_forward(
    data: pd.DataFrame,
    windows: list[Window],
    forecast_fn=seasonal_naive_forecast,
    season_length: int = 7,
) -> pd.DataFrame:
    """Evaluate a forecasting function over multiple chronological windows."""
    frame = data.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    results = []

    for i, window in enumerate(windows, start=1):
        train = frame[frame["date"] <= pd.Timestamp(window.train_end)].copy()
        actual = frame[
            (frame["date"] >= pd.Timestamp(window.forecast_start))
            & (frame["date"] <= pd.Timestamp(window.forecast_end))
        ].copy()

        if train.empty or actual.empty:
            continue

        pred = forecast_fn(train, actual, season_length=season_length)
        scores = evaluate(actual["sales"], pred)
        results.append(
            {
                "window": i,
                "train_end": window.train_end,
                "forecast_start": window.forecast_start,
                "forecast_end": window.forecast_end,
                **scores,
            }
        )

    return pd.DataFrame(results)


def summarize_windows(results: pd.DataFrame) -> pd.DataFrame:
    """Return mean and standard deviation across walk-forward windows."""
    metric_cols = [c for c in ("MAE", "RMSE", "WMAPE") if c in results.columns]
    summary = results[metric_cols].agg(["mean", "std"]).T.reset_index()
    summary.columns = ["metric", "mean", "std"]
    return summary
