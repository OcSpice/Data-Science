from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor=None

def build_models(seed=42):
    models={
      "Random Forest":RandomForestRegressor(n_estimators=150,max_depth=18,min_samples_leaf=2,n_jobs=-1,random_state=seed),
      "XGBoost":XGBRegressor(n_estimators=350,max_depth=8,learning_rate=.05,subsample=.85,colsample_bytree=.85,objective="reg:squarederror",n_jobs=-1,random_state=seed)
    }
    if LGBMRegressor:
      models["LightGBM"]=LGBMRegressor(n_estimators=350,learning_rate=.05,num_leaves=63,subsample=.85,colsample_bytree=.85,random_state=seed,verbosity=-1)
    return models

def save_model(model,path):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    joblib.dump(model,path)
