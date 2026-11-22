
from pathlib import Path
import pandas as pd

from src.data_pipeline import get_day_ahead_model_data
from src.models.knn import model_knn
from src.models.l2_regularizer import model_l2_ridge
from src.evaluate import eval_model

import numpy as np

from scipy.stats import loguniform
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor

# Create the tuning results directory
OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "tuning"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def export_knn_tuning(X_train, X_valid, y_train, y_valid):
    """Save validation results for every k value tested."""

    k_values = [1, 3, 5, 10, 15, 20, 30, 50, 75, 100]
    records = []

    for k in k_values:
        predictions, _ = model_knn(k, X_valid, X_train, y_train)
        mae, rmse = eval_model(y_valid, predictions)

        records.append({
            "k": k,
            "MAE": mae,
            "RMSE": rmse
        })

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_DIR / "knn_tuning.csv", index=False)

    print("\nkNN Tuning Results:")
    print(df.round(4).to_string(index=False))
    print(f"\nBest k: {df.loc[df['MAE'].idxmin(), 'k']}")


def export_ridge_tuning(X_train, X_valid, y_train, y_valid):
    """Save validation results for every Ridge alpha tested."""

    alpha_values = [0.01, 0.1, 1, 10, 100, 1000, 10000]
    records = []

    for alpha in alpha_values:
        predictions, _ = model_l2_ridge(
            alpha, X_valid, X_train, y_train
        )
        mae, rmse = eval_model(y_valid, predictions)

        records.append({
            "alpha": alpha,
            "MAE": mae,
            "RMSE": rmse
        })

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_DIR / "ridge_tuning.csv", index=False)

    print("\nRidge Tuning Results:")
    print(df.round(4).to_string(index=False))
    print(f"\nBest alpha: {df.loc[df['MAE'].idxmin(), 'alpha']}")


def export_decision_tree_tuning(X_train, y_train):
    """Export all Decision Tree grid-search configurations and CV scores."""

    param_grid = {
        "max_depth": [3, 5, 7, 10, 15, None],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 5, 10],
        "max_features": [None, "sqrt", 0.5],
        "ccp_alpha": [0.0, 0.001, 0.01],
    }

    search = GridSearchCV(
        DecisionTreeRegressor(random_state=42),
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )

    search.fit(X_train, y_train)

    # Export the full cross-validation results.
    df = pd.DataFrame(search.cv_results_)

    # Convert negative scoring values into positive errors.
    df["CV_MSE"] = -df["mean_test_score"]
    df["CV_RMSE"] = np.sqrt(df["CV_MSE"])

    # Keep the parameter settings and useful scoring information.
    columns = [
        "rank_test_score",
        "param_max_depth",
        "param_min_samples_split",
        "param_min_samples_leaf",
        "param_max_features",
        "param_ccp_alpha",
        "CV_MSE",
        "CV_RMSE",
        "std_test_score",
    ]

    df = df[columns].sort_values("rank_test_score")
    df.to_csv(OUTPUT_DIR / "decision_tree_tuning.csv", index=False)

    print("\nDecision Tree tuning saved.")
    print(f"Configurations tested: {len(df)}")
    print(f"Best parameters: {search.best_params_}")
    print(f"Best CV MSE: {-search.best_score_:.4f}")


def export_neural_network_tuning(X_train, y_train):
    """Export all Neural Network randomized-search configurations."""

    neural_network = MLPRegressor(
        max_iter=1000,
        early_stopping=True,
        n_iter_no_change=15,
        random_state=42,
    )

    input_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", neural_network),
    ])

    model = TransformedTargetRegressor(
        regressor=input_pipeline,
        transformer=StandardScaler(),
    )

    parameter_choices = {
        "regressor__mlp__hidden_layer_sizes": [
            (32,), (64,), (128,),
            (64, 32), (128, 64), (128, 64, 32),
        ],
        "regressor__mlp__alpha": loguniform(1e-5, 1e0),
        "regressor__mlp__learning_rate_init": loguniform(1e-4, 1e-2),
        "regressor__mlp__activation": ["relu", "tanh"],
        "regressor__mlp__batch_size": [32, 64, 128, 256],
    }

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=parameter_choices,
        n_iter=40,
        cv=5,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        random_state=42,
        refit=True,
    )

    search.fit(X_train, y_train)

    df = pd.DataFrame(search.cv_results_)

    # Convert negative MAE to positive MAE.
    df["CV_MAE"] = -df["mean_test_score"]

    columns = [
        "rank_test_score",
        "param_regressor__mlp__hidden_layer_sizes",
        "param_regressor__mlp__activation",
        "param_regressor__mlp__alpha",
        "param_regressor__mlp__batch_size",
        "param_regressor__mlp__learning_rate_init",
        "CV_MAE",
        "std_test_score",
    ]

    df = df[columns].sort_values("rank_test_score")
    df.to_csv(OUTPUT_DIR / "neural_network_tuning.csv", index=False)

    print("\nNeural Network tuning saved.")
    print(f"Configurations tested: {len(df)}")
    print(f"Best parameters: {search.best_params_}")
    print(f"Best CV MAE: {-search.best_score_:.4f}")


def main():
    print("Loading project data...")

    X_train, X_valid, _, y_train, y_valid, _ = (
        get_day_ahead_model_data()
    )

    # kNN and Ridge results were already exported.
    export_decision_tree_tuning(X_train, y_train)
    export_neural_network_tuning(X_train, y_train)

    print("\nAll tuning CSV files saved successfully.")


if __name__ == "__main__":
    main()