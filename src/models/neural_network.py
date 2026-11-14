from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from scipy.stats import loguniform      
from src.evaluate import eval_model  

def model_neural_network(X_valid, X_train, y_train):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            max_iter=1000,
            early_stopping=True,
            n_iter_no_change=15,
            random_state=42,
        )),
    ])

    # Scale the target too; MLPs train poorly on large-valued y
    model = TransformedTargetRegressor(regressor=pipe, transformer=StandardScaler())

    param_dist = {
        "regressor__mlp__hidden_layer_sizes": [(32,), (64,), (128,), (64, 32), (128, 64), (128, 64, 32)],
        "regressor__mlp__alpha": loguniform(1e-5, 1e0),
        "regressor__mlp__learning_rate_init": loguniform(1e-4, 1e-2),
        "regressor__mlp__activation": ["relu", "tanh"],
        "regressor__mlp__batch_size": [32, 64, 128, 256],
    }

    search = RandomizedSearchCV(
        model,
        param_dist,
        n_iter=40,
        cv=5,
        scoring="neg_mean_absolute_error",  # matches your MAE metric
        n_jobs=-1,
        random_state=42,
    )
    search.fit(X_train, y_train)

    print("Best params:", search.best_params_)
    print("CV MAE:", -search.best_score_)

    return search.predict(X_valid)