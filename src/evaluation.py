from src.naive_benchmark import *
from src.baseline_model import *

print(f"Naive Benchmark: MAE = {mae_naive:.3f} \t RMSE = {rmse_naive:.3f}")
print()
print(f"Linear Regression: MAE = {mae:.3f} \t RMSE = {rmse:.3f}")