import pandas as pd
import streamlit as st
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/"reports"/"model_comparison.csv"
st.set_page_config(page_title="Retail Sales Forecasting",layout="wide")
st.title("Retail Sales Forecasting & Demand Planning")
st.caption("Synthetic M5-like retail dataset — portfolio demonstration")
if REPORT.exists():
    df=pd.read_csv(REPORT)
    st.subheader("Model comparison")
    st.dataframe(df,use_container_width=True)
    metric=st.selectbox("Metric",["MAE","RMSE","WMAPE"])
    st.bar_chart(df.set_index("Model")[metric])
else:
    st.info("Run the forecasting pipeline first to generate reports/model_comparison.csv.")
