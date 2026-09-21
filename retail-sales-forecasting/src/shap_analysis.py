from pathlib import Path
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from .features import prepare_model_features

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "reports"
VIS = ROOT / "visuals"

def run():
    data = pd.concat([pd.read_csv(DATA / f"train_{year}.csv", low_memory=False)
                      for year in (2023, 2024, 2025)], ignore_index=True)
    X = prepare_model_features(data)
    y = data["sales"]
    mask = X.notna().all(axis=1)
    X, y = X.loc[mask], y.loc[mask]
    model = XGBRegressor(
        n_estimators=250, max_depth=6, learning_rate=.05, subsample=1,
        colsample_bytree=.85, min_child_weight=3, reg_lambda=2,
        n_jobs=-1, objective="reg:squarederror", random_state=42,
    )
    model.fit(X, np.log1p(y))
    sample = X.sample(min(2000, len(X)), random_state=42)
    values = shap.TreeExplainer(model)(sample)
    VIS.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    shap.plots.beeswarm(values, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(VIS / "08_shap_summary.png", dpi=160, bbox_inches="tight")
    plt.close()
    importance = pd.DataFrame({
        "feature": X.columns,
        "mean_abs_shap": np.abs(values.values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    importance.to_csv(OUT / "shap_feature_importance.csv", index=False)
    return importance

if __name__ == "__main__":
    print(run().head(15).to_string(index=False))
