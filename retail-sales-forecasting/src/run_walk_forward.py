"""Run three chronological 28-day walk-forward windows for the retail dataset.

Usage:
    python -m src.run_walk_forward
"""

from pathlib import Path

import pandas as pd

from .data import load_training, load_validation
from .walk_forward import make_windows, evaluate_walk_forward, summarize_windows


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def main() -> None:
    train = load_training()
    validation = load_validation()

    # Combine only data that would have been observable before the evaluation
    # windows. This script is intended for validation, not final test scoring.
    data = pd.concat([train, validation], ignore_index=True)

    windows = make_windows(
        first_train_end="2025-10-08",
        first_forecast_start="2025-10-09",
        n_windows=3,
        horizon_days=28,
        step_days=28,
    )

    results = evaluate_walk_forward(data, windows)
    summary = summarize_windows(results)

    REPORTS.mkdir(parents=True, exist_ok=True)
    results.to_csv(REPORTS / "walk_forward_windows.csv", index=False)
    summary.to_csv(REPORTS / "walk_forward_summary.csv", index=False)

    print("\nWalk-forward results")
    print(results.to_string(index=False))
    print("\nSummary")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
