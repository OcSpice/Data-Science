from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

st.set_page_config(page_title="Retail Demand Forecasting", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.block-container {padding-top:2rem; padding-bottom:3rem; max-width:1400px;}
.hero {padding:1.4rem 1.6rem; border-radius:18px; background:linear-gradient(135deg,#111827,#1f2937); color:white; margin-bottom:1.2rem;}
.hero h1 {font-size:2.25rem; margin-bottom:.35rem;}
.hero p {font-size:1.02rem; color:#d1d5db; margin-bottom:0;}
</style>
""", unsafe_allow_html=True)

def report(name):
    path = REPORTS / name
    return pd.read_csv(path) if path.exists() else None

def add_id_metadata(df):
    out = df.copy()
    if "id" in out:
        parts = out["id"].astype(str).str.split("_")
        out["cat_id"] = parts.str[0]
        out["dept_id"] = parts.str[0] + "_" + parts.str[1]
        out["item_id"] = parts.str[0] + "_" + parts.str[1] + "_" + parts.str[2]
        out["store_id"] = parts.str[-2] + "_" + parts.str[-1]
    return out

def load_forecast():
    path = REPORTS / "final_28_day_forecast_xgb_log_recursive.csv"
    if not path.exists():
        return None
    out = pd.read_csv(path)
    out["date"] = pd.to_datetime(out["date"])
    return add_id_metadata(out)

model_cmp = report("model_comparison.csv")
strategy = report("direct_vs_recursive_comparison.csv")
segments = report("volume_segment_performance.csv")
hybrid = report("hybrid_test_metrics.csv")
forecast = load_forecast()

st.markdown("""
<div class="hero">
<h1>Retail Demand Forecasting</h1>
<p>Multi-store, multi-category forecasting with validation discipline, explainability and operational error analysis.</p>
</div>
""", unsafe_allow_html=True)

best = model_cmp.loc[model_cmp.WMAPE.idxmin()] if model_cmp is not None else None
c1,c2,c3,c4 = st.columns(4)
c1.metric("Models evaluated", int(len(model_cmp)) if model_cmp is not None else "—")
c2.metric("Best validation WMAPE", f"{best.WMAPE:.1%}" if best is not None else "—")
c3.metric("Walk-forward WMAPE", "27.07%")
c4.metric("Hybrid test WMAPE", f"{hybrid.WMAPE.iloc[0]:.1%}" if hybrid is not None else "—")

section = st.sidebar.radio("Explore", ["Executive view","Forecast diagnostics","Model lab","Demand segmentation","Methodology"])

if section == "Executive view":
    st.subheader("What this project demonstrates")
    left,right = st.columns([1.1,.9])
    with left:
        st.markdown("""
**Forecasting workflow**

1. Exploratory demand analysis
2. Chronological walk-forward validation
3. Baseline vs ML model comparison
4. Recursive vs direct multi-step experiment
5. SHAP feature attribution
6. Volume-aware forecast policy
7. Store/category error diagnostics

The portfolio signal is not model complexity by itself. A simple Moving Average baseline remains visible, and every additional layer is evaluated against measured error.
""")
    with right:
        if model_cmp is not None:
            fig=px.bar(model_cmp.sort_values("WMAPE"),x="WMAPE",y="Model",orientation="h",title="Validation error by model",text_auto=".1%")
            fig.update_layout(xaxis_tickformat=".0%")
            st.plotly_chart(fig,use_container_width=True)
    wf=report("walk_forward_baseline.csv")
    if wf is not None:
        wf["window"]=wf.forecast_start.astype(str)+" → "+wf.forecast_end.astype(str)
        fig=px.line(wf,x="window",y="WMAPE",markers=True,title="Walk-forward WMAPE across validation windows")
        fig.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig,use_container_width=True)

elif section == "Forecast diagnostics":
    if forecast is None:
        st.warning("Run the forecasting pipeline to create reports/final_28_day_forecast_xgb_log_recursive.csv. The dashboard will then activate the interactive forecast view.")
    else:
        a,b,c=st.columns(3)
        store=a.selectbox("Store",["All"]+sorted(forecast.store_id.unique()))
        cat=b.selectbox("Category",["All"]+sorted(forecast.cat_id.unique()))
        horizon_options=["All"]+sorted(forecast.horizon.dropna().unique().tolist()) if "horizon" in forecast else ["All"]
        horizon=c.selectbox("Horizon",horizon_options)
        view=forecast.copy()
        if store!="All": view=view[view.store_id==store]
        if cat!="All": view=view[view.cat_id==cat]
        if horizon!="All": view=view[view.horizon==horizon]
        daily=view.groupby("date",as_index=False)[["sales","prediction"]].sum()
        daily["absolute_error"]=(daily.sales-daily.prediction).abs()
        wmape=daily.absolute_error.sum()/daily.sales.abs().sum()
        x,y=st.columns(2); x.metric("Selected-scope WMAPE",f"{wmape:.1%}"); y.metric("Forecast days",daily.date.nunique())
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=daily.date,y=daily.sales,mode="lines+markers",name="Actual"))
        fig.add_trace(go.Scatter(x=daily.date,y=daily.prediction,mode="lines+markers",name="Forecast"))
        fig.update_layout(title="Actual vs forecast",xaxis_title="Date",yaxis_title="Units",hovermode="x unified")
        st.plotly_chart(fig,use_container_width=True)
        st.plotly_chart(px.bar(daily,x="date",y="absolute_error",title="Daily absolute error"),use_container_width=True)

elif section == "Model lab":
    st.subheader("Direct vs recursive forecasting")
    if strategy is not None:
        fig=px.line(strategy,x="horizon",y="WMAPE",color="strategy",markers=True,title="Error by forecast horizon")
        fig.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig,use_container_width=True)
        st.dataframe(strategy.style.format({"MAE":"{:.3f}","RMSE":"{:.3f}","WMAPE":"{:.1%}"}),use_container_width=True,hide_index=True)
        st.caption("Controlled checkpoints: 1, 7, 14 and 28 days. Both strategies use the same feature set and XGBoost configuration.")
    st.info("The Moving Average baseline remains in the benchmark. Complexity is treated as an experiment that must earn its place through measured improvement.")

elif section == "Demand segmentation":
    st.subheader("Low-volume demand needs a different policy")
    if segments is not None:
        fig=px.bar(segments,x="volume_segment",y=["WMAPE_ML","WMAPE_policy"],barmode="group",title="ML forecast vs volume-aware policy")
        fig.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig,use_container_width=True)
        st.dataframe(segments.style.format({"MAE_ML":"{:.3f}","WMAPE_ML":"{:.1%}","MAE_policy":"{:.3f}","WMAPE_policy":"{:.1%}"}),use_container_width=True,hide_index=True)
    st.markdown("**Policy:** low-volume series use a 7-day moving-average fallback; genuinely intermittent series use a zero-demand rule; medium/high-volume series retain the ML forecast.")

else:
    st.subheader("Methodology")
    st.code("""data/
  train_2023.csv
  train_2024.csv
  train_2025.csv
  validation.csv
  test.csv
  test_actual.csv

src/
  features.py
  baselines.py
  modeling.py
  forecast_compare.py
  volume_segmentation.py
  shap_analysis.py

reports/
  model_comparison.csv
  walk_forward_baseline.csv
  direct_vs_recursive_comparison.csv
  volume_segment_performance.csv
  hybrid_test_metrics.csv
""",language="text")
    st.markdown("**Split:** chronological train → validation → held-out test.  **Horizon:** 28 days.  **Metrics:** MAE, RMSE, WMAPE.  **Explainability:** SHAP.  **Dataset:** synthetic M5/Walmart-style retail data for portfolio demonstration, not proprietary Walmart data.")
