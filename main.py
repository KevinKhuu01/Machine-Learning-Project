from src.data_pipeline import get_model_data
from src.naive_benchmark import mae_naive, rmse_naive
from src.baseline_model import mae, rmse


def main():
    # Load the model-ready training, validation, and test datasets
    X_train, X_valid, X_test, y_train, y_valid, y_test = get_model_data()

    print("Data pipeline completed successfully.")
    print("Training samples:", len(X_train))
    print("Validation samples:", len(X_valid))
    print("Test samples:", len(X_test))

    print("\nBaseline Results:")
    print(f"Naive Benchmark: MAE = {mae_naive:.3f} \t RMSE = {rmse_naive:.3f}")
    print(f"Linear Regression: MAE = {mae:.3f} \t RMSE = {rmse:.3f}")


if __name__ == "__main__":
    main()