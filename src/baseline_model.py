# Course-level Baseline Model
# Model is a simple linear regression model that will use the MAE and RMSE metrics to evaluate its performance. 
# The model will be trained on the training set and evaluated on the validation set. 
# The model will be used to predict the traffic volume for the next 15-minute interval;
# based on the current traffic volume and time based features.

# Imports
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.data_pipeline import get_model_data

linear_model = LinearRegression()

X_train, X_valid, X_test, y_train, y_valid, y_test = get_model_data()

linear_model.fit(X_train, y_train)

y_pred = linear_model.predict(X_valid)

mae = mean_absolute_error(y_valid, y_pred)
rmse = np.sqrt(mean_squared_error(y_valid, y_pred))

