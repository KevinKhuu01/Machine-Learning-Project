from src.data_pipeline import get_model_data


def main():
    # Load the model-ready training, validation, and test datasets
    X_train, X_valid, X_test, y_train, y_valid, y_test = get_model_data()

    print("Data pipeline completed successfully.")
    print("Training samples:", len(X_train))
    print("Validation samples:", len(X_valid))
    print("Test samples:", len(X_test))


if __name__ == "__main__":
    main()