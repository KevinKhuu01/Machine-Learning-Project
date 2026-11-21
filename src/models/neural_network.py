from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from scipy.stats import loguniform      
from src.evaluate import eval_model  
import numpy as np
import pandas as pd
from scipy.stats import t

def model_neural_network(X_valid, X_train, y_train):
    neural_network = MLPRegressor(max_iter= 1000, early_stopping=True, n_iter_no_change=15, random_state=42)

    # Scale input features before passing to network
    input_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", neural_network),
    ])

    # Scale target values during training
    model = TransformedTargetRegressor(regressor=input_pipeline, transformer=StandardScaler())

    parameter_choices = {
          "regressor__mlp__hidden_layer_sizes": [
            (32,), (64,), (128,), (64, 32), (128, 64), (128, 64, 32)
        ],
        "regressor__mlp__alpha": loguniform(1e-5, 1e0),
        "regressor__mlp__learning_rate_init": loguniform(1e-4, 1e-2),
        "regressor__mlp__activation": ["relu", "tanh"],
        "regressor__mlp__batch_size": [32, 64, 128, 256],
    }

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=parameter_choices,
        n_iter=40, # number of iterations to run through (number of different combinations to try)
        cv=5,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        random_state=42,
        refit=True, 
    )
    search.fit(X_train, y_train)

    # Print the selected settings and average cross-validation error
    best = search.best_params_
    print("\nNeural Network: Best Configuration")
    print(f"Hidden layers:         {best['regressor__mlp__hidden_layer_sizes']}")
    print(f"Activation:            {best['regressor__mlp__activation']}")
    print(f"L2 penalty (alpha):    {best['regressor__mlp__alpha']:.6f}")
    print(f"Batch size:            {best['regressor__mlp__batch_size']}")
    print(f"Initial learning rate: {best['regressor__mlp__learning_rate_init']:.8f}")
    print(f"Mean CV MAE (5 folds):  {-search.best_score_:.4f}")

    # Predict the validation targets, in the original vehicle-count units.
    y_preds = search.predict(X_valid)
    return y_preds



def run_neural_network_seeds(X_train, y_train, X_valid, y_valid):
    results = []
    actual = np.asarray(y_valid, dtype=float)

    print("Neural Network: 10 Seeds, Fixed Hyperparameters", flush=True)
    print("RMSE is pooled across all validation predictions.", flush=True)
    print(f"{'Seed':<8}{'MAE':>12}{'RMSE':>12}", flush=True)

    # Train a new network for each seed using the same data and settings.
    for seed in range(10):
        neural_network = MLPRegressor(
            # best parameters - results from tuning code above
            hidden_layer_sizes=(128, 64),
            activation="relu",
            alpha=0.1993893849355457,
            batch_size=128,
            learning_rate_init=0.0007961566078062948,
            # mlp parameters
            max_iter=1000,
            early_stopping=True,
            n_iter_no_change=15,
            random_state=seed,
        )

        # scale input
        input_pipeline = Pipeline([
            ("scaler", StandardScaler()),
                ("mlp", neural_network),
        ])

        # scale target
        model = TransformedTargetRegressor(
            regressor=input_pipeline,
            transformer=StandardScaler(),
        )

        # predict and find error
        model.fit(X_train, y_train) 
        predictions = model.predict(X_valid)
        mae, rmse = eval_model(actual, predictions)

        results.append({"seed": seed, "MAE": mae, "RMSE": rmse})
        print(f"{seed:<8}{mae:>12.4f}{rmse:>12.4f}", flush=True)

    runs = pd.DataFrame(results) 

    # print summary of runs
    summaries = []
    number_of_runs = len(runs)
    critical_value = t.ppf(0.975, df=number_of_runs - 1)

    print("\nSummary across seeds (approximate 95% CI for the mean)")
    for metric in ["MAE", "RMSE"]:
        mean = runs[metric].mean()
        standard_deviation = runs[metric].std(ddof=1)
        standard_error = standard_deviation / np.sqrt(number_of_runs)
        margin = critical_value * standard_error

        summaries.append({
            "metric": metric,
            "n_seeds": number_of_runs,
            "mean": mean,
            "std": standard_deviation,
            "ci95_lower": mean - margin,
            "ci95_upper": mean + margin,
        })
        print(
            f"{metric}: \nmean={mean:.4f}, SD={standard_deviation:.4f}"
            f"\n95% CI=[{mean - margin:.4f}, {mean + margin:.4f}]\n"
        )

    summary = pd.DataFrame(summaries)

    return runs, summary
