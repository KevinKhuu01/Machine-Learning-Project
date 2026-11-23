
import numpy as np
import pandas as pd

from main import build_models
from src.data_pipeline import get_day_ahead_model_data

# Load the same training, validation, and test data used in main.py
X_train, X_valid, X_test, y_train, y_valid, y_test = get_day_ahead_model_data()

# Generate predictions for all six models
validation_results, test_results, predictions, test_predictions = build_models(
    X_train, X_valid, X_test, y_train, y_valid, y_test
)

# Analyze test predictions instead of validation predictions
predictions = test_predictions

actual = y_test.to_numpy()

# Compare error statistics for every model
summary = []

for name, prediction in predictions.items():
    predicted = np.asarray(prediction)
    residuals = actual - predicted
    absolute_errors = np.abs(residuals)

    summary.append({
        "Model": name,
        "MAE": absolute_errors.mean(),
        "Median Absolute Error": np.median(absolute_errors),
        "90th Percentile Error": np.percentile(absolute_errors, 90),
        "Maximum Absolute Error": absolute_errors.max(),
        "Mean Residual": residuals.mean(),
        "Underprediction (%)": (residuals > 0).mean() * 100,
        "Overprediction (%)": (residuals < 0).mean() * 100,
    })

summary_df = pd.DataFrame(summary)

# Save results without changing the existing model code
summary_df.to_csv("results/error_analysis/test/error_summary.csv", index=False)

print("\nError Analysis Summary:")
print(summary_df.round(2).to_string(index=False))


# ---------------------------------------------------------------------------
# Analyze prediction errors by traffic volume.
#
# Group test observations into low, medium, and high traffic
# categories based on actual vehicle counts.
#
# For each model, calculate the MAE, mean residual, and percentage
# of predictions that underestimate the actual traffic volume.
# ---------------------------------------------------------------------------

traffic_groups = {
    "Low (0-25)": (actual >= 0) & (actual <= 25),
    "Medium (26-100)": (actual > 25) & (actual <= 100),
    "High (>100)": actual > 100,
}

volume_results = []

for name, prediction in predictions.items():
    predicted = np.asarray(prediction)
    residuals = actual - predicted

    for group_name, mask in traffic_groups.items():
        group_residuals = residuals[mask]

        if group_residuals.size == 0:
            continue

        volume_results.append({
            "Model": name,
            "Traffic Group": group_name,
            "Observations": group_residuals.size,
            "MAE": np.abs(group_residuals).mean(),
            "Mean Residual": group_residuals.mean(),
            "Underprediction (%)": (
                (group_residuals > 0).mean() * 100
            ),
        })

volume_df = pd.DataFrame(volume_results)

volume_df.to_csv(
    "results/error_analysis/test/error_by_volume.csv",
    index=False
)

print("\nError Analysis by Traffic Volume:")
print(volume_df.round(2).to_string(index=False))


# ---------------------------------------------------------------------------
# Analyze prediction errors by time of day.
#
# Calculate the MAE for each of the 32 predicted intervals
# between 9:00 AM and 5:00 PM.
# ---------------------------------------------------------------------------

interval_results = []

for name, prediction in predictions.items():
    predicted = np.asarray(prediction)
    absolute_errors = np.abs(actual - predicted)

    for interval in range(32):
        start_minutes = 9 * 60 + interval * 15
        hour, minute = divmod(start_minutes, 60)
        time_label = f"{hour:02d}:{minute:02d}"

        interval_results.append({
            "Model": name,
            "Time": time_label,
            "Interval": interval + 1,
            "MAE": absolute_errors[:, interval].mean(),
            "Mean Actual Volume": actual[:, interval].mean(),
            "Mean Residual": (
                actual[:, interval] - predicted[:, interval]
            ).mean(),
        })

interval_df = pd.DataFrame(interval_results)

interval_df.to_csv(
    "results/error_analysis/test/error_by_interval.csv",
    index=False
)

print("\nError Analysis by Time of Day:")
print(interval_df.round(2).to_string(index=False))


# ---------------------------------------------------------------------------
# Identify the largest prediction errors across all six models.
#
# Record the actual and predicted traffic volumes, forecast time,
# and error for each model's 10 most inaccurate predictions.
#
# Also identify observations that are difficult for multiple models.
# ---------------------------------------------------------------------------

largest_errors = []
all_absolute_errors = {}

for name, prediction in predictions.items():
    predicted = np.asarray(prediction)
    residuals = actual - predicted
    absolute_errors = np.abs(residuals)

    all_absolute_errors[name] = absolute_errors

    # Find the 10 largest errors across all test intervals.
    top_indices = np.argsort(
        absolute_errors,
        axis=None
    )[-10:][::-1]

    for flat_index in top_indices:
        sample_index, interval = np.unravel_index(
            flat_index,
            absolute_errors.shape
        )

        start_minutes = 9 * 60 + interval * 15
        hour, minute = divmod(start_minutes, 60)

        largest_errors.append({
            "Model": name,
            "Sample Index": sample_index,
            "Time": f"{hour:02d}:{minute:02d}",
            "Actual Volume": actual[sample_index, interval],
            "Predicted Volume": predicted[sample_index, interval],
            "Residual": residuals[sample_index, interval],
            "Absolute Error": absolute_errors[sample_index, interval],
        })

largest_errors_df = pd.DataFrame(largest_errors)

largest_errors_df.to_csv(
    "results/error_analysis/test/largest_errors.csv",
    index=False
)

print("\nTop 10 Largest Errors per Model:")
print(largest_errors_df.round(2).to_string(index=False))


# ---------------------------------------------------------------------------
# Identify observations with high errors across multiple models.
#
# Average the absolute errors of all six models for each observation.
# ---------------------------------------------------------------------------

average_absolute_errors = np.mean(
    np.stack(list(all_absolute_errors.values())),
    axis=0
)

top_shared_indices = np.argsort(
    average_absolute_errors,
    axis=None
)[-10:][::-1]

shared_errors = []

for flat_index in top_shared_indices:
    sample_index, interval = np.unravel_index(
        flat_index,
        average_absolute_errors.shape
    )

    start_minutes = 9 * 60 + interval * 15
    hour, minute = divmod(start_minutes, 60)

    row = {
        "Sample Index": sample_index,
        "Time": f"{hour:02d}:{minute:02d}",
        "Actual Volume": actual[sample_index, interval],
        "Average Absolute Error": average_absolute_errors[
            sample_index, interval
        ],
    }

    for name, errors in all_absolute_errors.items():
        row[name] = errors[sample_index, interval]

    shared_errors.append(row)

shared_errors_df = pd.DataFrame(shared_errors)

shared_errors_df.to_csv(
    "results/error_analysis/test/shared_large_errors.csv",
    index=False
)

print("\nLargest Shared Errors Across All Models:")
print(shared_errors_df.round(2).to_string(index=False))


