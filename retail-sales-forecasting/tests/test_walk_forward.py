import pandas as pd

from src.walk_forward import make_windows, seasonal_naive_forecast


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
