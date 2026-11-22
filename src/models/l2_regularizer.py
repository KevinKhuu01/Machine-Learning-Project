from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.evaluate import eval_model


def model_l2_ridge(a, X_valid, X_train, y_train, X_test=None):
    """
    Ridge (L2-regularized) linear regression.

    Preprocessing:
        StandardScaler on features, fit on X_train only (inside the pipeline).
        Scaling matters here because the L2 penalty is applied equally to all
        coefficients, so features need to be on the same scale.

    Parameters:
        a:       Ridge regularization strength (alpha). Larger = stronger shrinkage.
        X_valid: Features to predict.
        X_train: Training features.
        y_train: Training target.

    Returns:
        Predictions for X_valid (numpy array).
    """
    
    l2_model = make_pipeline(StandardScaler(), Ridge(alpha=a))
    l2_model.fit(X_train, y_train)

    y_valid = l2_model.predict(X_valid)
    y_test = l2_model.predict(X_test) if X_test is not None else None
    
    return y_valid, y_test

def tune_l2_alpha(X_train, X_valid, y_train, y_valid):
    best_a, best_mae = None, float("inf") # set best_mae to infinity to start. first iteration of loop will automatically be the best_mae
    for a in  [0.01, 0.1, 1, 10, 100, 1000, 10000]:
        y_pred, _ = model_l2_ridge(a, X_valid, X_train, y_train)
        mae, rmse = eval_model(y_valid, y_pred)
        if mae < best_mae:
            best_a, best_mae = a, mae

    print(f"Best alpha: {best_a}")
    print(f"Best MAE: {best_mae}")

    return best_a
