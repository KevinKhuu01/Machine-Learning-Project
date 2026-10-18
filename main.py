from src.data_pipeline import get_model_data
from src.naive_benchmark import mae_naive, rmse_naive
from src.baseline_model import mae, rmse


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
    print("\nBaseline Results:")
    print(f"\nNaive Benchmark: MAE = {mae_naive:.3f} \t RMSE = {rmse_naive:.3f}")
    print(f"\nLinear Regression: MAE = {mae:.3f} \t RMSE = {rmse:.3f}")
    print("\n-----------------------------------")


if __name__ == "__main__":
    main()