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

def build_models(X_train, X_valid, y_train, y_valid):
    best_k = tune_k(X_train, X_valid, y_train, y_valid)
    best_l2_alpha = tune_l2_alpha(X_train, X_valid, y_train, y_valid)

    preds = {
        "naive_benchmark": model_naive_benchmark(X_valid),
        "linear_regression": model_linear_regression(X_valid, X_train, y_train),
        "decision_tree": model_decision_tree(X_valid, X_train, y_train),
        "knn": model_knn(best_k, X_valid, X_train, y_train),
        "l2_regularization": model_l2_ridge(best_l2_alpha, X_valid, X_train, y_train),
        "neural_network": model_neural_network(X_valid, X_train, y_train),
    }
    results = {name: eval_model(y_valid, p) for name, p in preds.items()}

    return results, preds

def main():
    start = time.perf_counter()

    X_train, X_valid, X_test, y_train, y_valid, y_test = get_day_ahead_model_data()
    results, preds = build_models(X_train, X_valid, y_train, y_valid)    
         
    print("\n-----------------------------------")

    print("\nData pipeline completed successfully.")

    print("\n-----------------------------------")

    print("\nTraining samples:", len(X_train))
    print("\nValidation samples:", len(X_valid))
    print("\nTest samples:", len(X_test))

    print("\n-----------------------------------")

    print("\nMean vehicle volume per interval (training):", y_train.to_numpy().mean())
    print("\nMean vehicle volume per interval (validation):", y_valid.to_numpy().mean())

    print("\n-----------------------------------")

    print("\nResults:")
    for name, (mae, rmse) in results.items():
        print(f"\n{name}: \nMAE = {mae:.2f} \t RMSE = {rmse:.2f}")

    print("\n-----------------------------------")

    print("\nNeural Network Seeds:")

    run_neural_network_seeds(X_train, y_train, X_valid, y_valid)

    print("\n-----------------------------------")
    
    end = time.perf_counter()
    elapsed_time = end-start
    minutes_elapsed = (int)(elapsed_time // 60)
    seconds_elapsed =  elapsed_time % 60
    print(f"\n\rElapsed time: {minutes_elapsed} min {seconds_elapsed :.1f} sec", end="", flush=True)

    print("\n-----------------------------------")

    plot_all(y_valid, results, preds) 

if __name__ == "__main__":
    main()