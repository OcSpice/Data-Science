import pandas as pd

def seasonal_naive(train,future,lag=7):
    hist=train.sort_values(["id","date"]).set_index(["id","date"])["sales"]
    dates=pd.to_datetime(future["date"])-pd.Timedelta(days=lag)
    return pd.Series([hist.get((r.id,d),float("nan")) for r,d in zip(future.itertuples(),dates)],index=future.index)

def moving_average(train,future,window=7):
    hist=train.sort_values(["id","date"]).copy()
    hist["_ma"]=hist.groupby("id")["sales"].transform(lambda s:s.rolling(window).mean())
    return future["id"].map(hist.groupby("id")["_ma"].last())
