from pathlib import Path
import numpy as np,pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
DATA=Path(__file__).resolve().parents[1]/"data"; OUT=Path(__file__).resolve().parents[1]/"reports"; VIS=Path(__file__).resolve().parents[1]/"visuals"
def load():
 t=pd.concat([pd.read_csv(DATA/f"train_{y}.csv",low_memory=False) for y in (2023,2024,2025)],ignore_index=True);v=pd.read_csv(DATA/"validation.csv",low_memory=False)
 for x in (t,v):x["date"]=pd.to_datetime(x["date"])
 return pd.concat([t,v],ignore_index=True)
def score(y,p):
 y=np.asarray(y);p=np.asarray(p);return mean_absolute_error(y,p),np.sqrt(mean_squared_error(y,p)),np.abs(y-p).sum()/np.abs(y).sum()
def eda(d):
 VIS.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)
 specs=[("01_sales_trend","Daily Retail Sales Trend",d.groupby("date").sales.sum(), "Date","Units"),
 ("02_weekly_seasonality","Weekly Seasonality",d.groupby("wday").sales.mean(),"Weekday","Average units"),
 ("03_monthly_seasonality","Monthly Seasonality",d.groupby("month").sales.mean(),"Month","Average units"),
 ("04_store_comparison","Store Comparison",d.groupby("store_id").sales.sum(),"Store","Units"),
 ("05_category_comparison","Category Comparison",d.groupby("cat_id").sales.sum(),"Category","Units")]
 for fn,title,s,x,y in specs:
  fig,ax=plt.subplots(figsize=(10,5));ax.plot(s.index,s.values,marker="o") if s.index.dtype!="object" else ax.bar(s.index,s.values);ax.set(title=title,xlabel=x,ylabel=y);fig.tight_layout();fig.savefig(VIS/f"{fn}.png",dpi=150);plt.close(fig)
 q=d.groupby("sell_price").sales.mean().reset_index().dropna().query("sell_price>=0");fig,ax=plt.subplots(figsize=(9,5));ax.scatter(q.sell_price,q.sales,s=8,alpha=.35);ax.set(title="Price vs Average Sales",xlabel="Sell price",ylabel="Average sales");fig.tight_layout();fig.savefig(VIS/"06_price_vs_sales.png",dpi=150);plt.close(fig)
 e=d.assign(event=d.event_name_1.notna());s=e.groupby("event").sales.mean();fig,ax=plt.subplots(figsize=(8,5));ax.bar(["No event","Event"],[s.get(False,0),s.get(True,0)]);ax.set(title="Event Effect",ylabel="Average units");fig.tight_layout();fig.savefig(VIS/"07_event_effect.png",dpi=150);plt.close(fig)
 d.groupby("store_id").sales.agg(["sum","mean","std"]).to_csv(OUT/"store_eda_summary.csv");d.groupby("cat_id").sales.agg(["sum","mean","std"]).to_csv(OUT/"category_eda_summary.csv")
def walk_forward(d):
 rows=[]
 for start in pd.to_datetime(["2025-09-01","2025-10-06","2025-11-06"]):
  end=start+pd.Timedelta(days=27);h=d[d.date<start];f=d[(d.date>=start)&(d.date<=end)]
  lag=h[["id","date","sales"]].copy();lag["date"]=lag.date+pd.Timedelta(days=7);p=f.merge(lag.rename(columns={"sales":"pred"}),on=["id","date"],how="left").dropna(subset=["pred"]);a=score(p.sales,p.pred);rows.append({"forecast_start":start.date(),"forecast_end":end.date(),"MAE":a[0],"RMSE":a[1],"WMAPE":a[2],"n_obs":len(p)})
 o=pd.DataFrame(rows);o.to_csv(OUT/"walk_forward_baseline.csv",index=False);o.assign(stat="mean").groupby("stat")[["MAE","RMSE","WMAPE"]].mean().to_csv(OUT/"walk_forward_summary.csv")
 return o
if __name__=="__main__":
 d=load();eda(d);print(walk_forward(d))
