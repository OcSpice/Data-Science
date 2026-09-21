import numpy as np
from src.metrics import evaluate,wmape

def test_metrics_zero_error():
    y=np.array([10,20,30])
    assert evaluate(y,y)["MAE"]==0
    assert evaluate(y,y)["RMSE"]==0
    assert wmape(y,y)==0
