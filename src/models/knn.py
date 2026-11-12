from sklearn.neighbors import KNeighborsRegressor
from src.evaluate import eval_model

def model_knn(k, X_valid, X_train, y_train):
    knn_model = KNeighborsRegressor(k)
    knn_model.fit(X_train, y_train)
    y_pred = knn_model.predict(X_valid)

    return y_pred

def tune_k(X_train, X_valid, y_train, y_valid):
    best_k, best_mae = None, float("inf") # set best_mae to infinity to start. first iteration of loop will automatically be the best_mae
    for k in  [1, 3, 5, 10, 15, 20, 30, 50, 75, 100]:
        mae, _ = eval_model(y_valid, model_knn(k, X_valid, X_train, y_train))
        if mae < best_mae:
            best_k, best_mae = k, mae

    return best_k


