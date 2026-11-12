from src.data_pipeline import get_day_ahead_model_data
from src.models.naive_benchmark import *
from src.models.linear_regression import *
from src.models.decision_tree import *
from src.evaluate import eval_model

def build_models(X_train, X_valid, y_train, y_valid):
    results = {}
    results["naive_benchmark"] = eval_model(y_valid, model_naive_benchmark(X_valid))
    results["linear_regression"] = eval_model(y_valid, model_linear_regression(X_valid, X_train, y_train))
    results["decision_tree"] = eval_model(y_valid, model_decision_tree(X_valid, X_train, y_train))

    return results

def main():
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


if __name__ == "__main__":
    main()