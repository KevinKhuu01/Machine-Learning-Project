from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# evaluates model mae and rmse
def eval_model(target, prediction):  
      mae = mean_absolute_error(target, prediction)
      rmse = np.sqrt(mean_squared_error(target, prediction))
      return mae, rmse