# Naive Benchmark 
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error 
from data_pipeline import get_model_data

X_train, X_valid, X_test, y_train, y_valid, y_test = get_model_data()

y_pred_naive = X_test["volume_15min"].to_numpy()  # Use the last observed volume as the prediction for the next time step

mae_naive = mean_absolute_error(y_test, y_pred_naive)
rmse_naive = np.sqrt(mean_squared_error(y_test, y_pred_naive)) 

print("Results of Naive Benchmark Calculation: \n")
print(f"Naive Benchmark MAE: {mae_naive:.3f}")
print(f"Naive Benchmark RMSE: {rmse_naive:.3f}")
