# Naive Benchmark 
def model_naive_benchmark(X_valid):
    # Use the previous day's 32 volumes to predict the next day's volumes
    volume_features = [f"previous_volume_{i}" for i in range(32)]
  
    return X_valid[volume_features].copy()



