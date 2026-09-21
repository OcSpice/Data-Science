"""Direct multi-step XGBoost forecasting.
For each horizon h, train on historical rows whose target is sales h days ahead,
then predict exactly the h-th future day from the common forecast origin.
"""
import numpy as np,pandas as pd
from xgboost import XGBRegressor
from .features import prepare_features

def direct_forecast(history,future,horizons=range(1,29)):
    history=history.sort_values(["id","date"]).copy()
    X=prepare_features(history); y0=history["sales"]
    out=[]
    for h in horizons:
        y=history.groupby("id")["sales"].shift(-h)
        mask=y.notna() & X.notna().all(axis=1)
        model=XGBRegressor(n_estimators=80,max_depth=4,learning_rate=.06,subsample=.9,colsample_bytree=.9,min_child_weight=4,reg_lambda=3,n_jobs=-1,objective="reg:squarederror",random_state=42)
        model.fit(X.loc[mask],np.log1p(y.loc[mask]))
        target_date=pd.Timestamp(sorted(future["date"].unique())[h-1])
        day=future[future["date"].eq(target_date)].copy()
        z=pd.concat([history,day.assign(sales=np.nan)],ignore_index=True)
        xf=prepare_features(z).tail(len(day)).reindex(columns=X.columns,fill_value=0)
        day["prediction"]=np.clip(np.expm1(model.predict(xf)),0,None);day["horizon"]=h
        out.append(day[["id","date","prediction","horizon"]])
    return pd.concat(out,ignore_index=True)
