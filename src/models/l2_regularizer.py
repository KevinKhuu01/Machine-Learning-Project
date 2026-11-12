from sklearn.linear_model import Ridge
from src.evaluate import eval_model

def model_l2_ridge(a, X_valid, X_train, y_train):
    l2_model = Ridge(alpha=a)
    l2_model.fit(X_train, y_train)
    y_pred = l2_model.predict(X_valid)

    return y_pred

def tune_l2_alpha(X_train, X_valid, y_train, y_valid):
    best_a, best_mae = None, float("inf") # set best_mae to infinity to start. first iteration of loop will automatically be the best_mae
    for a in  [0.01, 0.1, 1, 10, 100, 1000, 10000]:
        mae, _ = eval_model(y_valid, model_l2_ridge(a, X_valid, X_train, y_train))
        if mae < best_mae:
            best_a, best_mae = a, mae

    return best_a
