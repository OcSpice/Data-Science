from pathlib import Path
import pandas as pd
import streamlit as st
ROOT=Path(__file__).resolve().parents[1];REPORT=ROOT/"reports";VIS=ROOT/"visuals"
st.set_page_config(page_title="Retail Demand Forecasting",layout="wide")
st.title("Retail Sales Forecasting & Demand Planning")
st.caption("Sales → Forecast → Error → Store → Category → Business implication")
mc=REPORT/"model_comparison.csv"
if mc.exists():
 m=pd.read_csv(mc); c=st.columns(4)
 for i,(lab,val) in enumerate([("Best validation MAE",m.MAE.min()),("Best validation RMSE",m.RMSE.min()),("Best validation WMAPE",m.WMAPE.min()),("Models tested",len(m))]): c[i].metric(lab,f"{val:.3f}" if i<3 else str(val))
 st.subheader("Model comparison");st.dataframe(m,use_container_width=True)
fp=REPORT/"final_28_day_forecast_xgb_log_recursive.csv"
if fp.exists():
 f=pd.read_csv(fp);meta=pd.read_csv(ROOT/"data"/"test.csv",usecols=["id","cat_id","store_id"]);f=f.merge(meta,on="id",how="left")
 st.subheader("28-day forecast");a,b=st.columns(2);store=a.selectbox("Store",["All"]+sorted(f.store_id.unique()));cat=b.selectbox("Category",["All"]+sorted(f.cat_id.unique()));v=f.copy()
 if store!="All":v=v[v.store_id==store]
 if cat!="All":v=v[v.cat_id==cat]
 d=v.groupby("date")[["sales","prediction"]].sum();c1,c2=st.columns(2);c1.line_chart(d);c2.line_chart((d.sales-d.prediction).abs());st.metric("Selected-scope WMAPE",f"{(d.sales-d.prediction).abs().sum()/d.sales.abs().sum():.1%}")
st.subheader("EDA and explainability")
for n,t in [("01_sales_trend.png","Sales trend"),("02_weekly_seasonality.png","Weekly seasonality"),("03_monthly_seasonality.png","Monthly seasonality"),("04_store_comparison.png","Store comparison"),("05_category_comparison.png","Category comparison"),("06_price_vs_sales.png","Price vs sales"),("07_event_effect.png","Event effects"),("08_shap_summary.png","SHAP explainability"),("09_actual_vs_predicted_28d.png","Actual vs predicted"),("10_horizon_error_28d.png","Forecast horizon error")]:
 p=VIS/n
 if p.exists():st.image(str(p),caption=t,use_container_width=True)
st.subheader("Business implication")
st.markdown("- Use forecasts for short-term replenishment and inventory planning.\n- Inspect error by store/category before operational use.\n- Keep simple baselines visible; complexity must earn its place through measured improvement.\n- Monitor low-volume and event-driven series separately.")
