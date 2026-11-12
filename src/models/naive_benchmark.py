# Naive Benchmark 
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error 
from src.data_pipeline import get_day_ahead_model_data
from src.evaluate import eval_model

def model_naive_benchmark():
    X_train, X_valid, X_test, y_train, y_valid, y_test = get_day_ahead_model_data()

    # Use the previous day's 32 volumes to predict the next day's volumes
    volume_features = [f"previous_volume_{i}" for i in range(32)]
    y_pred_naive = X_valid[volume_features].copy()

    # Match the prediction column names to the target columns
    y_pred_naive.columns = y_valid.columns

    return y_valid, y_pred_naive

def naive_error():
    y_valid, y_pred = model_naive_benchmark()
    return eval_model(y_valid, y_pred)


