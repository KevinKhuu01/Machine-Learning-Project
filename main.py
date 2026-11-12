from src.data_pipeline import get_day_ahead_model_data
from src.models.naive_benchmark import *
from src.models.linear_regression import *
from src.models.decision_tree import *
from src.models.knn import *
from src.models.l2_regularizer import *
from src.models.neural_network import *
from src.evaluate import eval_model
import time

def build_models(X_train, X_valid, y_train, y_valid):
    best_k = tune_k(X_train, X_valid, y_train, y_valid)
    best_l2_alpha = tune_l2_alpha(X_train, X_valid, y_train, y_valid)
    best_nn_alpha = tune_nn_alpha(X_train, X_valid, y_train, y_valid)
    results = {}

    results["naive_benchmark"] = eval_model(y_valid, model_naive_benchmark(X_valid))
    results["linear_regression"] = eval_model(y_valid, model_linear_regression(X_valid, X_train, y_train))
    results["decision_tree"] = eval_model(y_valid, model_decision_tree(X_valid, X_train, y_train))
    results["knn"] = eval_model(y_valid, model_knn(best_k, X_valid, X_train, y_train))
    results["l2_regularization"] = eval_model(y_valid, model_l2_ridge(best_l2_alpha, X_valid, X_train, y_train))
    results["neural_network"] = eval_model(y_valid, model_neural_network(best_nn_alpha, X_valid, X_train, y_train))

    return results

def main():
    start = time.perf_counter() 
    X_train, X_valid, X_test, y_train, y_valid, y_test = get_day_ahead_model_data()
    results = build_models(X_train, X_valid, y_train, y_valid)

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

    end = time.perf_counter()
    elapsed_time = end-start
    minutes_elapsed = (int)(elapsed_time // 60)
    seconds_elapsed =  elapsed_time % 60
    print(f"\nElapsed time: {minutes_elapsed} min {seconds_elapsed :.1f} sec")

    print("\n-----------------------------------")

if __name__ == "__main__":
    main()