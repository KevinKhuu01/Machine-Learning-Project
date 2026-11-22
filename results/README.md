# Results Package

This directory contains the experimental results and supporting evidence for Deliverable #5.

## Model Comparison

**File:** `model_comparison/validation_results.csv`

Contains the validation MAE and RMSE for all six forecasting models.

- Dataset: Validation set
- Samples: 1,988 day-ahead samples
- Predictions: 32 intervals per sample
- Source: `main.py`, `build_models()` and `src/evaluate.py`
- Run ID: `VAL-MAIN-01`
- Note: Values are rounded to two decimal places and were recorded from the previously completed validation experiment. The run ID is a package reference assigned retrospectively, not an original experiment identifier.

## Error Analysis

**Directory:** `error_analysis/`

Contains the error summary, traffic-volume analysis, time-of-day analysis, largest prediction errors, and observations with large errors across multiple models.

- Source: `error_analysis.py`
- Dataset: Validation set
- Run ID: Not recorded for the original execution

## Additional Experimental Results

The following directories organize the supporting results from the experiments conducted throughout the project:

- `per_seed/` — Contains the Neural Network's validation results across 10 random seeds (0–9).
  - `neural_network_seeds.csv`: Individual MAE and RMSE for each seed.
  - `neural_network_summary.csv`: Mean, standard deviation, and 95% confidence intervals.
  - Source: `run_neural_network_seeds()` in `src/models/neural_network.py`, called from `main.py`.
  - Dataset: Validation set.
  - Run ID: `VAL-NN-SEEDS-01` (assigned for the results package).
- `tuning/` — Contains the hyperparameter configurations tested and their evaluation scores.
  - `knn_tuning.csv`: Validation MAE and RMSE for 10 values of k. Best k = 20.
  - `ridge_tuning.csv`: Validation MAE and RMSE for 7 regularization strengths. Best alpha = 100.
  - `decision_tree_tuning.csv`: Results for 864 Decision Tree configurations using 5-fold cross-validation. Best CV MSE = 235.4283.
  - `neural_network_tuning.csv`: Results for 40 Neural Network configurations using 5-fold cross-validation. Best CV MAE = 6.5694.
  - Source: `export_tuning.py`, using the existing model implementations and tuning configurations.
  - Run IDs: `VAL-KNN-TUNE-01`, `VAL-RIDGE-TUNE-01`, `CV-DT-TUNE-01`, and `CV-NN-TUNE-01` (assigned for the results package).
  - Note: kNN and Ridge were evaluated on the validation set, while Decision Tree and Neural Network used cross-validation on the training set.
### Sensitivity Analysis

The `sensitivity/` directory contains results showing how model performance changes when individual hyperparameters are varied.

- `decision_tree_depth.csv`: Validation MAE across seven tree depths.
- `knn_neighbors.csv`: Validation MAE across six values of k.
- `../tuning/ridge_tuning.csv`: Validation MAE across seven Ridge regularization strengths.

Source: Deliverable #5 Report, Section 8 (Experiment Log), experiments 6–8.

The Decision Tree and kNN sensitivity results were transcribed from the experiment log. The Ridge results were exported from the project's tuning script.

Run references: `SENS-DT-01`, `SENS-KNN-01`, and `VAL-RIDGE-TUNE-01`. These identifiers were assigned for package organization and are not original execution IDs.
### Robustness Analysis

The `robustness/` directory contains results evaluating how model performance changes under input perturbations.

- `noise_robustness.csv`: Documented validation MAE for Ridge, Decision Tree, kNN, and Neural Network with clean inputs and Gaussian noise at 0.5 times the training feature standard deviation.
- `noise_robustness_summary.csv`: Reported percentage increases in validation MAE at noise level 0.1.
- `../per_seed/neural_network_seeds.csv`: Individual validation MAE and RMSE results across 10 Neural Network random seeds.
- `../per_seed/neural_network_summary.csv`: Mean, standard deviation, and 95% confidence intervals across those seeds.

Source: Deliverable #5 Report, Sections 6 and 8, experiments 9–10, and `main.py` for the newly exported seed results.

The Gaussian-noise experiment perturbed previous-day volume features in the validation inputs without retraining. It used one noise draw with random seed 0.

Exact validation MAE values for noise levels 0.1 and 0.25 were not available in the report and have not been reconstructed.

Run references: `ROB-NOISE-01` (report-transcribed results) and `VAL-NN-SEEDS-01` (newly exported seed results). These identifiers were assigned for package organization and are not original execution IDs.
### Ablation Analysis

The `ablation/` directory contains results evaluating the importance of individual features and feature groups.

- `previous_day_volumes.csv`: Validation MAE after removing all 32 previous-day traffic-volume features from Ridge, kNN, and Neural Network models.
- `feature_groups.csv`: Changes in validation MAE after removing individual time blocks, day of week, location, or direction features.
- `knn_scaling.csv`: Validation MAE for kNN with raw versus standardized features, using k = 5.

Source: Deliverable #5 Report, Section 8 (Experiment Log), experiments 11–19.

Positive MAE changes indicate worse performance after feature removal. Negative changes indicate improved performance.

The evening traffic-volume block had the greatest individual contribution across all three models. Removing all previous-day volume features caused substantial performance degradation.

Run references: `ABL-VOLUME-01`, `ABL-GROUPS-01`, and `ABL-KNN-SCALE-01`. These are organizational identifiers assigned for the results package, not original execution IDs.

All values were transcribed from the report's experiment log. Original raw experiment outputs and reproduction scripts for these ablations have not been independently verified.

## Figures

The following figures are generated by `src/visualize.py` through `plot_all()` in `main.py`. All figures use validation-set predictions.

1. **`../figures/metric_comparison.png`** — Comparison of MAE and RMSE across all six forecasting models. Source: `plot_metric_comparison()`.
2. **`../figures/pred_vs_actual.png`** — Predicted versus actual vehicle volumes for each model, including a reference line representing perfect predictions. Source: `plot_pred_vs_actual()`.
3. **`../figures/time_series.png`** — Actual vehicle volumes compared with Naive Benchmark, Decision Tree, and Neural Network predictions across the first 500 flattened validation intervals. Source: `plot_time_series()`.
4. **`../figures/residuals.png`** — Residual distributions (actual minus predicted) for all six models, displayed as boxplots with outliers hidden. Source: `plot_residuals()`.

Source code: `src/visualize.py`

Dataset: Validation set.

Run reference: `VAL-MAIN-01` (retrospectively assigned package identifier; original figure-generation run IDs were not recorded).

These figures can be regenerated by running `python main.py` from the project root after installing dependencies and preparing the required data.