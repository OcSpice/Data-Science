from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

st.set_page_config(page_title="Demand Planning", page_icon="📦", layout="wide")
st.title("Advanced Demand Planning & Inventory Simulation")
st.caption("Retail Sales Forecasting • synthetic M5-style dataset")

required = [
    "demand_segment_summary.csv",
    "uncertainty_summary.csv",
    "inventory_scenarios.csv",
    "business_exposure.csv",
]
missing = [name for name in required if not (REPORTS / name).exists()]

if missing:
    st.warning("Planning reports have not been generated yet.")
    st.code("python -m advanced_demand_planning.src.run_planning")
    st.write("Missing: " + ", ".join(missing))
    st.stop()

segments = pd.read_csv(REPORTS / "demand_segment_summary.csv")
uncertainty = pd.read_csv(REPORTS / "uncertainty_summary.csv")
scenarios = pd.read_csv(REPORTS / "inventory_scenarios.csv")
exposure = pd.read_csv(REPORTS / "business_exposure.csv")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Item-store series", f"{uncertainty.shape[0]:,}")
c2.metric("Base reorder requirement", f"{exposure.loc[exposure.scenario.eq('Base demand'), 'total_reorder_requirement'].iloc[0]:,.0f}")
c3.metric("Base stockout-risk flags", f"{int(exposure.loc[exposure.scenario.eq('Base demand'), 'stockout_risk_count'].iloc[0]):,}")
c4.metric("Planning service level", "95%")

st.subheader("Demand segmentation")
st.dataframe(segments, use_container_width=True, hide_index=True)

st.subheader("Inventory scenario exposure")
st.dataframe(exposure, use_container_width=True, hide_index=True)

st.subheader("Reorder-point distribution")
st.bar_chart(
    scenarios.groupby("scenario", as_index=True)["reorder_point"].mean()
)

st.subheader("Uncertainty profile")
u1, u2, u3 = st.columns(3)
u1.metric("Median MAE", f"{uncertainty.mae.median():.2f}")
u2.metric("Median error SD", f"{uncertainty.error_std.median():.2f}")
u3.metric("Median P95 abs. error", f"{uncertainty.error_p95_abs.median():.2f}")

st.info(
    "Inventory position is simulated as seven days of recent baseline demand. "
    "The uncertainty measure is a historical 7-day seasonal-naive error proxy, "
    "not a calibrated probabilistic forecast. The current dataset has no observed "
    "inventory, supplier lead-time, purchase-order, or stockout records."
)
