import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def mae(y_true,y_pred): return float(mean_absolute_error(y_true,y_pred))
def rmse(y_true,y_pred): return float(np.sqrt(mean_squared_error(y_true,y_pred)))
def wmape(y_true,y_pred):
    y=np.asarray(y_true); p=np.asarray(y_pred)
    denom=np.abs(y).sum()
    return float(np.abs(y-p).sum()/denom) if denom else np.nan
def evaluate(y_true,y_pred):
    return {"MAE":mae(y_true,y_pred),"RMSE":rmse(y_true,y_pred),"WMAPE":wmape(y_true,y_pred)}
