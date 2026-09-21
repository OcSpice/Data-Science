"""Direct multi-step XGBoost forecasting.

This module is retained as a reusable experiment. For portfolio benchmarking,
use forecast_compare.py, which evaluates selected horizons with identical
settings against the recursive strategy.
"""
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from .features import prepare_model_features

def direct_forecast(history, future, horizons=(1, 7, 14, 28)):
    history = history.sort_values(["id", "date"]).copy()
    X = prepare_model_features(history)
    outputs = []
    for horizon in horizons:
        target = history.groupby("id", sort=False)["sales"].shift(-horizon)
        mask = target.notna() & X.notna().all(axis=1)
        model = XGBRegressor(
            n_estimators=120, max_depth=5, learning_rate=.06,
            subsample=.9, colsample_bytree=.9, min_child_weight=4,
            reg_lambda=3, n_jobs=-1, objective="reg:squarederror",
            random_state=42,
        )
        model.fit(X.loc[mask], np.log1p(target.loc[mask]))
        target_date = pd.Timestamp(sorted(future["date"].unique())[horizon - 1])
        day = future[future["date"].eq(target_date)].copy().sort_values("id")
        combined = pd.concat([history, day.assign(sales=np.nan)], ignore_index=True)
        xf = prepare_model_features(combined)
        keys = pd.MultiIndex.from_frame(combined[["id", "date"]])
        wanted = pd.MultiIndex.from_frame(day[["id", "date"]])
        xf = xf.loc[keys.isin(wanted)].reindex(columns=X.columns, fill_value=0)
        day["prediction"] = np.clip(np.expm1(model.predict(xf)), 0, None)
        day["horizon"] = horizon
        outputs.append(day[["id", "date", "prediction", "horizon"]])
    return pd.concat(outputs, ignore_index=True)
