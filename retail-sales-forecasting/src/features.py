import numpy as np
import pandas as pd

LAGS = (1, 7, 14, 28)
ROLLING_WINDOWS = (7, 14, 28)

def add_calendar_features(df):
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["dow_sin"] = np.sin(2 * np.pi * out["wday"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["wday"] / 7)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    return out

def add_lag_features(df):
    out = df.sort_values(["id", "date"]).copy()
    if "sales" not in out.columns:
        return out
    group = out.groupby("id", sort=False)["sales"]
    for lag in LAGS:
        out[f"lag_{lag}"] = group.shift(lag)
    shifted = group.shift(1)
    for window in ROLLING_WINDOWS:
        out[f"rolling_mean_{window}"] = shifted.rolling(window).mean().reset_index(level=0, drop=True)
    out["rolling_std_28"] = shifted.rolling(28).std().reset_index(level=0, drop=True)
    return out

def prepare_features(df, include_target=True):
    out = add_lag_features(add_calendar_features(df))
    categorical = ["store_id", "state_id", "item_id", "dept_id", "cat_id"]
    out = pd.get_dummies(out, columns=categorical, drop_first=False, dtype=int)
    drop = ["id", "d", "date", "weekday", "event_name_1", "event_type_1",
            "event_name_2", "event_type_2", "wm_yr_wk"]
    if not include_target:
        drop.append("sales")
    return out.drop(columns=[c for c in drop if c in out.columns])

def prepare_model_features(df):
    """Feature matrix for modeling; the sales target is never returned as a feature."""
    return prepare_features(df, include_target=False)
