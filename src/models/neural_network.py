from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline 
from sklearn.preprocessing import StandardScaler       
from src.evaluate import eval_model  

def model_neural_network(a, X_valid, X_train, y_train):
    nn_model =make_pipeline(StandardScaler(), MLPRegressor(alpha=a, max_iter=500, random_state=42)) 
    nn_model.fit(X_train, y_train)
    y_pred = nn_model.predict(X_valid)

    return y_pred

def tune_nn_alpha(X_train, X_valid, y_train, y_valid):
    best_alpha, best_mae = None, float("inf")
    for a in [0.0001, 0.001, 0.01, 0.1, 1]:
        mae, _ = eval_model(y_valid, model_neural_network(a, X_valid, X_train, y_train))
        if mae < best_mae:
            best_alpha, best_mae = a, mae

    return best_alpha
