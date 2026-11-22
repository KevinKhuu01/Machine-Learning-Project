from src.data_pipeline import get_day_ahead_model_data
from src.models.naive_benchmark import *
from src.models.linear_regression import *
from src.models.decision_tree import *
from src.models.knn import *
from src.models.l2_regularizer import *
from src.models.neural_network import *
from src.evaluate import eval_model
from src.visualize import plot_all
import time
import threading
import pandas as pd

def build_models(X_train, X_valid, X_test, y_train, y_valid, y_test):
    best_k = tune_k(X_train, X_valid, y_train, y_valid)
    best_l2_alpha = tune_l2_alpha(X_train, X_valid, y_train, y_valid)

    preds = {
        "naive_benchmark": model_naive_benchmark(X_valid, X_test),
        "linear_regression": model_linear_regression(X_valid, X_train, y_train, X_test),
        "decision_tree": model_decision_tree(X_valid, X_train, y_train, X_test),
        "knn": model_knn(best_k, X_valid, X_train, y_train, X_test),
        "l2_regularization": model_l2_ridge(best_l2_alpha, X_valid, X_train, y_train, X_test),
        "neural_network": model_neural_network(X_valid, X_train, y_train, X_test),
    }

    valid_preds = {name: outputs[0] for name, outputs in preds.items()}
    test_preds = {name: outputs[1] for name, outputs in preds.items()}

    validation_results = {name: eval_model(y_valid, prediction) for name, prediction in valid_preds.items()}
    test_results = {name: eval_model(y_test, prediction) for name, prediction in test_preds.items()}

    comparison_valid_df = pd.DataFrame([
        {"Model": name, "MAE": mae, "RMSE": rmse, "Dataset": "Validation"}
        for name, (mae, rmse) in validation_results.items()
    ])

    comparison_test_df = pd.DataFrame([
            {"Model": name, "MAE": mae, "RMSE": rmse, "Dataset": "Test"}
            for name, (mae, rmse) in test_results.items()
        ])

    comparison_valid_df.to_csv(
        "results/model_comparison/validation_results.csv",
        index=False
    )
    comparison_test_df.to_csv(
        "results/model_comparison/test_results.csv",
        index=False
    )

    return validation_results, test_results, valid_preds, test_preds


def main():
    start = time.perf_counter()

    X_train, X_valid, X_test, y_train, y_valid, y_test = get_day_ahead_model_data()
    valid_results, test_results, valid_preds, test_preds = build_models(X_train, X_valid, X_test, y_train, y_valid, y_test)

    print("\nValidation comparison CSV saved successfully.")
         
    print("\n-----------------------------------")

    print("\nData pipeline completed successfully.")

    print("\n-----------------------------------")

    print("\nTraining samples:", len(X_train))
    print("\nValidation samples:", len(X_valid))
    print("\nTest samples:", len(X_test))

    print("\n-----------------------------------")

    print("\nMean vehicle volume per interval (training):", y_train.to_numpy().mean())
    print("\nMean vehicle volume per interval (validation):", y_valid.to_numpy().mean())
    print("\nMean vehicle volume per interval (test):", y_test.to_numpy().mean())

    print("\n-----------------------------------")

    print("\nValidation results:")
    for name, (mae, rmse) in valid_results.items():
        print(f"\n{name}: \nMAE = {mae:.2f} \t RMSE = {rmse:.2f}")

    print("\n-----------------------------------")

    print("\nTest results:")
    for name, (mae, rmse) in test_results.items():
        print(f"\n{name}: \nMAE = {mae:.2f} \t RMSE = {rmse:.2f}")

    print("\n-----------------------------------")

    print("\nNeural Network Seeds:")

    runs, summary = run_neural_network_seeds(X_train, y_train, X_valid, y_valid, X_test, y_test)
    
    runs.to_csv("results/per_seed/neural_network_seeds.csv", index=False)
    
    summary.to_csv("results/per_seed/neural_network_summary.csv", index=False)

    print("\nNeural Network seed results saved successfully.")

    print("\n-----------------------------------")
    
    end = time.perf_counter()
    elapsed_time = end-start
    minutes_elapsed = (int)(elapsed_time // 60)
    seconds_elapsed =  elapsed_time % 60
    print(f"\n\rElapsed time: {minutes_elapsed} min {seconds_elapsed :.1f} sec", end="", flush=True)

    print("\n-----------------------------------")

    plot_all(y_valid, valid_results, valid_preds, set_name="validation")
    plot_all(y_test, test_results, test_preds, set_name="test")


if __name__ == "__main__":
    main()