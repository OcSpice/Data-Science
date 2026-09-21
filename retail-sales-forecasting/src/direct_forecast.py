"""Direct multi-step XGBoost scaffold. One model is fitted per horizon; use on validation before productionizing."""
from pathlib import Path
import numpy as np,pandas as pd
from xgboost import XGBRegressor
from features import prepare_features
def direct_forecast(history,forecast,horizons=range(1,29)):
 history=history.sort_values(["id","date"]).copy();X=prepare_features(history);cols=X.columns
 out=[]
 for h in horizons:
  y=history.groupby("id").sales.shift(-h);m=y.notna()&X.notna().all(1);model=XGBRegressor(n_estimators=80,max_depth=4,learning_rate=.06,subsample=.9,colsample_bytree=.9,min_child_weight=4,reg_lambda=3,n_jobs=-1,objective="reg:squarederror",random_state=42);model.fit(X.loc[m],np.log1p(y.loc[m]))
  z=pd.concat([history,forecast.assign(sales=np.nan)],ignore_index=True);xf=prepare_features(z).tail(len(forecast)).reindex(columns=cols,fill_value=0);p=np.clip(np.expm1(model.predict(xf)),0,None);q=forecast[["id","date"]].copy();q["prediction"]=p;q["horizon"]=h;out.append(q)
 return pd.concat(out,ignore_index=True)
