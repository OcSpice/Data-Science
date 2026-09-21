import pandas as pd

from src.walk_forward import (
    Window,
    make_windows,
    seasonal_naive_forecast,
    summarize_windows,
)


def test_make_windows_are_chronological():
    windows = make_windows(
        "2025-10-08",
        "2025-10-09",
        n_windows=3,
        horizon_days=28,
        step_days=28,
    )
    assert len(windows) == 3
    assert windows[0].forecast_end < windows[1].forecast_start
    assert windows[1].forecast_end < windows[2].forecast_start


def test_seasonal_naive_uses_previous_week():
    train = pd.DataFrame(
        {
            "id": ["A"] * 8,
            "date": pd.date_range("2025-01-01", periods=8),
            "sales": list(range(1, 9)),
        }
    )
    future = pd.DataFrame(
        {
            "id": ["A"],
            "date": [pd.Timestamp("2025-01-09")],
            "sales": [99],
        }
    )
    pred = seasonal_naive_forecast(train, future, season_length=7)
    assert pred.iloc[0] == 2


def test_summarize_model_windows():
    results = pd.DataFrame(
        {
            "model": ["Moving Average", "Moving Average", "Seasonal Naive"],
            "MAE": [1.0, 3.0, 2.0],
            "RMSE": [2.0, 4.0, 3.0],
            "WMAPE": [0.10, 0.30, 0.20],
        }
    )
    summary = summarize_windows(results)
    moving = summary.loc[summary["model"] == "Moving Average"].iloc[0]

    assert moving["MAE_mean"] == 2.0
    assert moving["WMAPE_mean"] == 0.20


def test_explicit_walk_forward_window_has_no_overlap():
    windows = [
        Window("2025-08-31", "2025-09-01", "2025-09-28"),
        Window("2025-10-05", "2025-10-06", "2025-11-02"),
        Window("2025-11-05", "2025-11-06", "2025-12-03"),
    ]
    assert windows[0].forecast_end < windows[1].forecast_start
    assert windows[1].forecast_end < windows[2].forecast_start
