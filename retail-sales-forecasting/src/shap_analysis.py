from pathlib import Path
import sys,numpy as np,pandas as pd,shap,matplotlib.pyplot as plt
from xgboost import XGBRegressor
sys.path.insert(0,str(Path(__file__).resolve().parent));from features import prepare_features
DATA=Path(__file__).resolve().parents[1]/"data";ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"reports";VIS=ROOT/"visuals"
d=pd.concat([pd.read_csv(DATA/f"train_{y}.csv",low_memory=False) for y in (2023,2024,2025)],ignore_index=True)
X=prepare_features(d).dropna();y=d.loc[X.index,"sales"]
m=XGBRegressor(n_estimators=250,max_depth=6,learning_rate=.05,subsample=1,colsample_bytree=.85,min_child_weight=3,reg_lambda=2,n_jobs=-1,objective="reg:squarederror",random_state=42);m.fit(X,np.log1p(y))
sample=X.sample(min(2000,len(X)),random_state=42);sv=shap.TreeExplainer(m)(sample);VIS.mkdir(exist_ok=True);OUT.mkdir(exist_ok=True)
shap.plots.beeswarm(sv,max_display=15,show=False);plt.tight_layout();plt.savefig(VIS/"08_shap_summary.png",dpi=160,bbox_inches="tight");plt.close()
pd.DataFrame({"feature":X.columns,"mean_abs_shap":np.abs(sv.values).mean(0)}).sort_values("mean_abs_shap",ascending=False).to_csv(OUT/"shap_feature_importance.csv",index=False)
