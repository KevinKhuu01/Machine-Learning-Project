# Naive Benchmark 
def model_naive_benchmark(X_valid, X_test):
    # Use the previous day's 32 volumes to predict the next day's volumes
    volume_features = [f"previous_volume_{i}" for i in range(32)]

    # predictions
    y_valid = X_valid[volume_features].to_numpy()
    y_test = X_test[volume_features].to_numpy()
    
    return y_valid, y_test



