"""
Leakage-safe end-to-end pipeline for Time Series Sales Forecasting.

Workflow:
1. Train on 2023-01-01 through 2025-11-05.
2. Select Holt-Winters smoothing parameters on the 2025-11-06 through
   2025-12-03 validation period.
3. Refit the selected model on train + validation data.
4. Evaluate the locked model on the final 2025-12-04 through 2025-12-31
   test period using test_actual.csv.

The final test period is never used for parameter selection.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from data_loader import DataLoader
from evaluation import BaselineModels, EvaluationMetrics
from holt_winters import HoltWintersModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TRAIN_END = pd.Timestamp("2025-11-05")
VALIDATION_START = pd.Timestamp("2025-11-06")
VALIDATION_END = pd.Timestamp("2025-12-03")
TEST_START = pd.Timestamp("2025-12-04")
TEST_END = pd.Timestamp("2025-12-31")

PARAM_GRID = {
    "alpha": [0.1, 0.2, 0.3, 0.5, 0.7],
    "beta": [0.05, 0.1, 0.2, 0.3],
    "gamma": [0.05, 0.1, 0.2, 0.3],
}


def aggregate_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate item/store observations to daily total sales."""
    required = {"date", "sales"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    out = (
        df.assign(date=pd.to_datetime(df["date"]))
        .groupby("date", as_index=False)["sales"]
        .sum()
        .rename(columns={"sales": "total_sales"})
        .sort_values("date")
        .reset_index(drop=True)
    )
    return out


def load_test_actual(data_dir: Path) -> pd.DataFrame:
    """Load and aggregate the held-out final test actuals."""
    path = data_dir / "test_actual.csv"
    if not path.exists():
        raise FileNotFoundError(f"Final test actuals not found: {path}")
    return aggregate_sales(pd.read_csv(path))


def split_train_validation(
    merged: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create the documented chronological train/validation split."""
    daily = aggregate_sales(merged)
    train = daily[daily["date"] <= TRAIN_END].copy()
    validation = daily[
        (daily["date"] >= VALIDATION_START) & (daily["date"] <= VALIDATION_END)
    ].copy()

    if train.empty or validation.empty:
        raise ValueError("Train or validation period is empty.")
    if train["date"].max() >= validation["date"].min():
        raise ValueError("Train and validation periods overlap.")
    return train, validation


def select_parameters(
    train_sales: np.ndarray,
    validation_sales: np.ndarray,
    season_length: int = 7,
    seasonal_type: str = "add",
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Select smoothing parameters using validation MAPE only.

    Each candidate is fitted only on training observations, then forecast
    across the validation horizon. The final test period is not touched.
    """
    rows: List[Dict[str, float]] = []

    for alpha in PARAM_GRID["alpha"]:
        for beta in PARAM_GRID["beta"]:
            for gamma in PARAM_GRID["gamma"]:
                model = HoltWintersModel(
                    season_length=season_length,
                    seasonal_type=seasonal_type,
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma,
                )
                model.fit(train_sales)
                forecast = np.maximum(model.predict(len(validation_sales)), 0)
                rows.append(
                    {
                        "alpha": alpha,
                        "beta": beta,
                        "gamma": gamma,
                        "MAPE": float(EvaluationMetrics.mape(validation_sales, forecast)),
                        "MAE": float(EvaluationMetrics.mae(validation_sales, forecast)),
                        "RMSE": float(EvaluationMetrics.rmse(validation_sales, forecast)),
                    }
                )

    results = pd.DataFrame(rows).sort_values(
        ["MAPE", "MAE", "RMSE"], ascending=True
    ).reset_index(drop=True)
    best = results.iloc[0]
    params = {
        "alpha": float(best["alpha"]),
        "beta": float(best["beta"]),
        "gamma": float(best["gamma"]),
    }
    return params, results


def evaluate_forecast(
    actual: np.ndarray,
    predicted: np.ndarray,
    training_data: np.ndarray,
) -> Dict[str, float]:
    """Return the project's core forecasting metrics."""
    return {
        "MAPE": float(EvaluationMetrics.mape(actual, predicted)),
        "MAE": float(EvaluationMetrics.mae(actual, predicted)),
        "RMSE": float(EvaluationMetrics.rmse(actual, predicted)),
        "SMAPE": float(EvaluationMetrics.smape(actual, predicted)),
        "MASE": float(EvaluationMetrics.mase(actual, predicted, training_data)),
    }


def baseline_forecasts(
    training_data: np.ndarray,
    steps: int,
) -> Dict[str, np.ndarray]:
    """Generate non-ML benchmark forecasts from the available history."""
    return {
        "Naive": BaselineModels.naive_forecast(training_data, steps),
        "Seasonal_Naive": BaselineModels.seasonal_naive_forecast(
            training_data, steps, season_length=7
        ),
        "Mean": BaselineModels.mean_forecast(training_data, steps),
        "Drift": BaselineModels.drift_forecast(training_data, steps),
    }


def run_pipeline(
    data_dir: str = "Datasets",
    output_dir: str = "reports",
) -> Dict:
    """Run validation-based model selection and final test evaluation."""
    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(str(data_path))
    loader.load_train_files()
    loader.load_calendar()
    loader.load_sell_prices()
    loader.load_validation()

    merged = loader.merge_data(include_validation=True)
    train, validation = split_train_validation(merged)
    test = load_test_actual(data_path)

    if test["date"].min() != TEST_START or test["date"].max() != TEST_END:
        raise ValueError(
            f"Unexpected final test range: {test['date'].min().date()} to "
            f"{test['date'].max().date()}"
        )

    train_sales = train["total_sales"].to_numpy(dtype=float)
    validation_sales = validation["total_sales"].to_numpy(dtype=float)
    test_sales = test["total_sales"].to_numpy(dtype=float)

    params, tuning = select_parameters(train_sales, validation_sales)

    # Validation score of the selected configuration.
    validation_model = HoltWintersModel(season_length=7, **params)
    validation_model.fit(train_sales)
    validation_pred = np.maximum(
        validation_model.predict(len(validation_sales)), 0
    )
    validation_metrics = evaluate_forecast(
        validation_sales, validation_pred, train_sales
    )

    # Lock parameters, then refit using all information available before test.
    final_history = np.concatenate([train_sales, validation_sales])
    final_model = HoltWintersModel(season_length=7, **params)
    final_model.fit(final_history)
    test_pred = np.maximum(final_model.predict(len(test_sales)), 0)
    test_metrics = evaluate_forecast(test_sales, test_pred, final_history)

    baseline_pred = baseline_forecasts(final_history, len(test_sales))
    baseline_metrics = {
        name: evaluate_forecast(test_sales, pred, final_history)
        for name, pred in baseline_pred.items()
    }

    predictions = pd.DataFrame(
        {
            "date": test["date"],
            "actual_sales": test_sales,
            "holt_winters_forecast": test_pred,
        }
    )
    predictions.to_csv(out_path / "forecast_predictions.csv", index=False)
    tuning.to_csv(out_path / "parameter_search.csv", index=False)

    metrics = {
        "validation": validation_metrics,
        "final_test": test_metrics,
        "final_test_baselines": baseline_metrics,
    }
    (out_path / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_path / "model_parameters.json").write_text(
        json.dumps(
            {
                "season_length": 7,
                "seasonal_type": "add",
                "selection_metric": "validation MAPE",
                "parameters": params,
                "train_end": str(TRAIN_END.date()),
                "validation_start": str(VALIDATION_START.date()),
                "validation_end": str(VALIDATION_END.date()),
                "test_start": str(TEST_START.date()),
                "test_end": str(TEST_END.date()),
            },
            indent=2,
        )
    )

    summary = {
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "selected_parameters": params,
        "validation_metrics": validation_metrics,
        "final_test_metrics": test_metrics,
        "final_test_baselines": baseline_metrics,
    }
    (out_path / "executive_summary.txt").write_text(
        "TIME SERIES SALES FORECASTING - EXECUTIVE SUMMARY\n"
        "=================================================\n\n"
        f"Selected parameters: {params}\n"
        f"Validation MAPE: {validation_metrics['MAPE']:.2f}%\n"
        f"Final test MAPE: {test_metrics['MAPE']:.2f}%\n"
        f"Final test MAE: {test_metrics['MAE']:.2f}\n"
        f"Final test RMSE: {test_metrics['RMSE']:.2f}\n\n"
        "Parameters were selected on the validation period only. "
        "The final test period was held out until after parameter selection.\n"
    )

    logger.info("Validation metrics: %s", validation_metrics)
    logger.info("Final test metrics: %s", test_metrics)
    logger.info("Selected parameters: %s", params)
    return summary


if __name__ == "__main__":
    result = run_pipeline()
    print(json.dumps(result, indent=2))
