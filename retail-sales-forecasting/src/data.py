from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

def load_training():
    return pd.concat([pd.read_csv(DATA / f"train_{y}.csv", low_memory=False) for y in (2023, 2024, 2025)], ignore_index=True)

def load_validation():
    return pd.read_csv(DATA / "validation.csv")

def load_test():
    return pd.read_csv(DATA / "test.csv")

def load_test_actual():
    return pd.read_csv(DATA / "test_actual.csv")
