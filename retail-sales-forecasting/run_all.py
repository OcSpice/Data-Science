"""Run the main reproducible retail forecasting and demand-planning pipeline.

Usage:
    python run_all.py

Run this file from the retail-sales-forecasting directory.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


STEPS = [
    ("flagship EDA and baseline analysis", [sys.executable, "-m", "src.flagship_analysis"]),
    ("walk-forward validation", [sys.executable, "-m", "src.run_walk_forward"]),
    ("advanced demand planning", [sys.executable, "-m", "advanced_demand_planning.src.run_planning"]),
]


def main() -> None:
    for name, command in STEPS:
        print(f"\n{'=' * 72}\nRunning: {name}\n{'=' * 72}")
        subprocess.run(command, cwd=ROOT, check=True)

    print("\nPipeline completed successfully.")
    print("Generated/updated reports are under reports/.")


if __name__ == "__main__":
    main()
