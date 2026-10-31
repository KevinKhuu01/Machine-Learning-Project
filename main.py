from src.data_pipeline import get_model_data
from src.models.naive_benchmark import *
from src.models.linear_regression import *
from src.models.decision_tree import *

def main():
    # Load the model-ready training, validation, and test datasets
    X_train, X_valid, X_test, y_train, y_valid, y_test = get_model_data()

    print("\n-----------------------------------")

    print("\nData pipeline completed successfully.")

    print("\n-----------------------------------")

    print("\nTraining samples:", len(X_train))
    print("\nValidation samples:", len(X_valid))
    print("\nTest samples:", len(X_test))

    print("\n-----------------------------------")

    print("\nMean vehicle volume per interval (training): ", X_train["volume_15min"].mean())
    print("\nMean vehicle volume per interval (validation): ", X_valid["volume_15min"].mean())

    print("\n-----------------------------------")
    naive_mae, naive_rmse = naive_error()
    lin_mse, lin_rmse = linear_regression_error()
    dt_mae, dt_rmse = decision_tree_error()

    print("\nResults:")
    print(f"\nNaive Benchmark: \nMAE = {naive_mae:.3f} \t RMSE = {naive_rmse:.3f}")
    print(f"\n\nLinear Regression: \nMAE = {lin_mse:.3f} \t RMSE = {lin_rmse:.3f}")
    print(f"\n\nDecision Tree: \nMAE = {dt_mae:.3f} \t RMSE = {dt_rmse:.3f}")
    
    print("\n-----------------------------------")


if __name__ == "__main__":
    main()