from sklearn.tree import DecisionTreeRegressor
from src.data_pipeline import get_model_data
from src.evaluate import eval_model

def model_decision_tree():
    model = DecisionTreeRegressor()
    X_train, X_valid, X_test, y_train, y_valid, y_test = get_model_data()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)
    
    return y_valid, y_pred

def decision_tree_error():
    y_valid, y_pred = model_decision_tree()
    return eval_model(y_pred, y_valid)