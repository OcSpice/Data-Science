"""Controlled direct-vs-recursive comparison at selected forecast horizons.

The comparison intentionally uses the same feature set, forecast origin, target
transformation, and XGBoost configuration. Training 28 separate direct models
is expensive on this dataset, so the portfolio benchmark focuses on horizons
1, 7, 14, and 28 days—the operational checkpoints used in the dashboard.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from .features import prepare_model_features

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
HORIZONS = (1, 7, 14, 28)

def load_history():
    parts = [pd.read_csv(DATA / f"train_{year}.csv", low_memory=False) for year in (2023, 2024, 2025)]
    history = pd.concat(parts, ignore_index=True)
    history["date"] = pd.to_datetime(history["date"])
    return history

def load_validation():
    df = pd.read_csv(DATA / "validation.csv", low_memory=False)
    df["date"] = pd.to_datetime(df["date"])
    return df

def model():
    return XGBRegressor(
        n_estimators=120, max_depth=5, learning_rate=0.06,
        subsample=0.9, colsample_bytree=0.9, min_child_weight=4,
        reg_lambda=3, objective="reg:squarederror", n_jobs=-1, random_state=42
    )

def score(actual, predicted):
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "WMAPE": np.abs(actual - predicted).sum() / np.abs(actual).sum(),
    }

def _future_features(history, day, columns):
    combined = pd.concat([history, day.assign(sales=np.nan)], ignore_index=True)
    features = prepare_model_features(combined)
    keys = pd.MultiIndex.from_frame(combined[["id", "date"]])
    wanted = pd.MultiIndex.from_frame(day[["id", "date"]])
    selected = features.loc[keys.isin(wanted)].copy()
    return selected.reindex(columns=columns, fill_value=0)

def direct_forecast(history, validation, horizons=HORIZONS):
    features = prepare_model_features(history)
    outputs = []
    for horizon in horizons:
        target = history.groupby("id", sort=False)["sales"].shift(-horizon)
        mask = target.notna() & features.notna().all(axis=1)
        fitted = model()
        fitted.fit(features.loc[mask], np.log1p(target.loc[mask]))
        target_date = validation["date"].min() + pd.Timedelta(days=horizon - 1)
        day = validation[validation["date"].eq(target_date)].copy().sort_values("id")
        x_future = _future_features(history, day, features.columns)
        prediction = np.clip(np.expm1(fitted.predict(x_future)), 0, None)
        scored = day[["id", "date", "sales"]].copy()
        scored["prediction"] = prediction
        scored["horizon"] = horizon
        scored["strategy"] = "Direct"
        outputs.append(scored)
    return pd.concat(outputs, ignore_index=True)

def recursive_forecast(history, validation, horizons=HORIZONS):
    features = prepare_model_features(history)
    mask = features.notna().all(axis=1)
    fitted = model()
    fitted.fit(features.loc[mask], np.log1p(history.loc[mask, "sales"]))
    state = history.copy()
    outputs = []
    max_h = max(horizons)
    for step in range(1, max_h + 1):
        target_date = validation["date"].min() + pd.Timedelta(days=step - 1)
        day = validation[validation["date"].eq(target_date)].copy().sort_values("id")
        x_future = _future_features(state, day, features.columns)
        prediction = np.clip(np.expm1(fitted.predict(x_future)), 0, None)
        day["prediction"] = prediction
        day["horizon"] = step
        day["strategy"] = "Recursive"
        outputs.append(day[["id", "date", "sales", "prediction", "horizon", "strategy"]])
        state = pd.concat([state, day.assign(sales=prediction)], ignore_index=True)
    return pd.concat(outputs, ignore_index=True).query("horizon in @horizons")

def run():
    history = load_history()
    validation = load_validation()
    direct = direct_forecast(history, validation)
    recursive = recursive_forecast(history, validation)
    scored = pd.concat([direct, recursive], ignore_index=True)
    rows = []
    for (strategy, horizon), group in scored.groupby(["strategy", "horizon"]):
        metrics = score(group.sales, group.prediction)
        rows.append({"strategy": strategy, "horizon": horizon, **metrics, "n_obs": len(group)})
    report = pd.DataFrame(rows).sort_values(["horizon", "strategy"])
    REPORTS.mkdir(exist_ok=True)
    report.to_csv(REPORTS / "direct_vs_recursive_comparison.csv", index=False)
    scored.to_csv(REPORTS / "direct_vs_recursive_predictions.csv", index=False)
    return report

if __name__ == "__main__":
    print(run().to_string(index=False))
