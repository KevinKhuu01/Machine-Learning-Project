# Naive Benchmark 
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error 
from src.data_pipeline import get_model_data
from src.evaluate import eval_model

def model_naive_benchmark():
    X_train, X_valid, X_test, y_train, y_valid, y_test = get_model_data()
    y_pred_naive = X_valid["volume_15min"].shift(1)  # Use the last observed volume as the prediction for the next time step

    y_valid_naive = y_valid.iloc[1:]  # Exclude the first value since it has no previous value to compare
    y_pred_naive = y_pred_naive.iloc[1:]  # Align predictions with the target values

    return y_valid_naive, y_pred_naive
    
def naive_error():
    y_valid, y_pred = model_naive_benchmark()
    return eval_model(y_pred, y_valid)


