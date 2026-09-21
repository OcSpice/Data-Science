"""Walk-forward evaluation utilities for daily retail forecasting.

The evaluator keeps each validation window strictly chronological:
fit -> forecast the next horizon -> score -> move the cutoff forward.
It supports transparent baselines and recursive ML forecasts so future actual
sales are never used as lag features during a multi-day forecast.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .baselines import moving_average
from .features import prepare_model_features
from .metrics import evaluate
from .modeling import build_models


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
        train_end += pd.Timedelta(days=step_days)
        forecast_start += pd.Timedelta(days=step_days)

    return windows


def seasonal_naive_forecast(
    train: pd.DataFrame,
    forecast: pd.DataFrame,
    season_length: int = 7,
) -> pd.Series:
    """Forecast each item-store series using the last seasonal observation."""
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


def recursive_model_forecast(
    model,
    train: pd.DataFrame,
    future: pd.DataFrame,
) -> pd.Series:
    """Forecast a multi-day horizon recursively without future-sales leakage.

    The model is fit once on the historical training window. Each future day is
    predicted in sequence, and that prediction is then added to the history
    used to construct later lag and rolling features.
    """
    history = train.copy()
    history["date"] = pd.to_datetime(history["date"])
    future = future.copy()
    future["date"] = pd.to_datetime(future["date"])

    train_features = prepare_model_features(history)
    model_columns = train_features.columns.tolist()
    model.fit(train_features, history["sales"])

    predictions = []
    for day in sorted(future["date"].unique()):
        day_rows = future[future["date"] == day].copy()
        day_rows["sales"] = np.nan

        history_window = history[
            history["date"] >= day - pd.Timedelta(days=28)
        ].copy()
        combined = pd.concat([history_window, day_rows], ignore_index=True)
        features = prepare_model_features(combined)
        day_features = features.iloc[-len(day_rows):].reindex(columns=model_columns)

        day_prediction = np.maximum(model.predict(day_features), 0)
        day_rows["sales"] = day_prediction
        predictions.append(day_rows[["id", "date", "sales"]].copy())
        history = pd.concat([history, day_rows], ignore_index=True)

    predicted = pd.concat(predictions, ignore_index=True)
    return future[["id", "date"]].merge(
        predicted, on=["id", "date"], how="left"
    )["sales"].set_axis(future.index)


def evaluate_model_walk_forward(
    data: pd.DataFrame,
    windows: list[Window],
    model_names: tuple[str, ...] = ("XGBoost", "LightGBM"),
    seed: int = 42,
) -> pd.DataFrame:
    """Evaluate baselines and selected ML models across all walk-forward windows."""
    frame = data.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    rows = []

    for i, window in enumerate(windows, start=1):
        train = frame[frame["date"] <= pd.Timestamp(window.train_end)].copy()
        actual = frame[
            (frame["date"] >= pd.Timestamp(window.forecast_start))
            & (frame["date"] <= pd.Timestamp(window.forecast_end))
        ].copy()
        if train.empty or actual.empty:
            continue

        baseline_predictions = {
            "Seasonal Naive": seasonal_naive_forecast(train, actual),
            "Moving Average": moving_average(train, actual),
        }
        for name, prediction in baseline_predictions.items():
            scores = evaluate(actual["sales"], prediction)
            rows.append({
                "window": i,
                "train_end": window.train_end,
                "forecast_start": window.forecast_start,
                "forecast_end": window.forecast_end,
                "model": name,
                **scores,
            })

        models = build_models(seed=seed)
        for name in model_names:
            if name not in models:
                continue
            prediction = recursive_model_forecast(models[name], train, actual)
            scores = evaluate(actual["sales"], prediction)
            rows.append({
                "window": i,
                "train_end": window.train_end,
                "forecast_start": window.forecast_start,
                "forecast_end": window.forecast_end,
                "model": name,
                **scores,
            })

    return pd.DataFrame(rows)


def summarize_windows(results: pd.DataFrame) -> pd.DataFrame:
    """Return mean and standard deviation across walk-forward windows."""
    metric_cols = [c for c in ("MAE", "RMSE", "WMAPE") if c in results.columns]
    if "model" in results.columns:
        return (
            results.groupby("model")
            .agg(
                MAE_mean=("MAE", "mean"),
                MAE_std=("MAE", "std"),
                RMSE_mean=("RMSE", "mean"),
                RMSE_std=("RMSE", "std"),
                WMAPE_mean=("WMAPE", "mean"),
                WMAPE_std=("WMAPE", "std"),
            )
            .reset_index()
        )

    summary = results[metric_cols].agg(["mean", "std"]).T.reset_index()
    summary.columns = ["metric", "mean", "std"]
    return summary
