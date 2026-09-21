"""Run multi-model chronological walk-forward validation.

Usage:
    python -m src.run_walk_forward
"""

from pathlib import Path

import pandas as pd

from .data import load_training, load_validation
from .walk_forward import Window, evaluate_model_walk_forward, summarize_windows

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def main() -> None:
    train = load_training()
    validation = load_validation()
    data = pd.concat([train, validation], ignore_index=True)

    # These are the same three chronological 28-day windows used in the
    # flagship analysis. Each ML model is fit only on data before the window
    # and then forecasts recursively, so future actual sales are not used as
    # lag features.
    windows = [
        Window("2025-08-31", "2025-09-01", "2025-09-28"),
        Window("2025-10-05", "2025-10-06", "2025-11-02"),
        Window("2025-11-05", "2025-11-06", "2025-12-03"),
    ]

    results = evaluate_model_walk_forward(data, windows)
    summary = summarize_windows(results)

    REPORTS.mkdir(parents=True, exist_ok=True)
    results.to_csv(REPORTS / "walk_forward_windows.csv", index=False)
    summary.to_csv(REPORTS / "walk_forward_summary.csv", index=False)

    print("\nMulti-model walk-forward results")
    print(results.to_string(index=False))
    print("\nSummary")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
