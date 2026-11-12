# Course-level Baseline Model
# Model is a simple linear regression model that will use the MAE and RMSE metrics to evaluate its performance. 
# The model will be trained on the training set and evaluated on the validation set. 
# The model will be used to predict the traffic volume for the next 15-minute interval;
# based on the current traffic volume and time based features.

# Imports
from sklearn.linear_model import LinearRegression

def model_linear_regression(X_valid, X_train, y_train):
        linear_model = LinearRegression()
        linear_model.fit(X_train, y_train)
        y_pred = linear_model.predict(X_valid)

        return y_pred
