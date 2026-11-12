# Course-level Baseline Model
# Model is a simple linear regression model that will use the MAE and RMSE metrics to evaluate its performance. 
# The model will be trained on the training set and evaluated on the validation set. 
# The model will be used to predict the traffic volume for the next 15-minute interval;
# based on the current traffic volume and time based features.

# Imports
from sklearn.linear_model import LinearRegression
from src.data_pipeline import get_day_ahead_model_data
from src.evaluate import eval_model

def model_linear():
        # build models
        linear_model = LinearRegression()
        X_train, X_valid, X_test, y_train, y_valid, y_test = get_day_ahead_model_data()

        # train models 
        linear_model.fit(X_train, y_train)
        y_pred = linear_model.predict(X_valid)

        return y_valid, y_pred

def linear_regression_error():
    y_valid, y_pred = model_linear()
    return eval_model(y_pred, y_valid)
