# Machine-Learning Project - Toronto Traffic Volume Forecasting
## Project Setup

### 1. Clone the Repository

Clone the repository and navigate to the project directory.

### 2. Create a Virtual Environment

Create a Python virtual environment inside the project:

```bash
python3 -m venv .venv
```

### 3. Activate the Virtual Environment

#### macOS / Linux

```bash
source .venv/bin/activate
```

#### Windows

```bash
.venv\Scripts\activate
```

### 4. Install Project Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### Adding New Dependencies

If you install a new Python package or library for the project, update `requirements.txt` before committing your changes:

```bash
pip freeze > requirements.txt
```

This ensures that all team members can install the same project dependencies using:

```bash
pip install -r requirements.txt
```

### 5. Jupyter Notebooks

The exploratory data analysis notebooks are located in the `notebooks/` directory.

When running the notebooks, select the Python interpreter/kernel from the project's `.venv` virtual environment.

### Project Structure

```text
Machine-Learning-Project/
├── data/
│   ├── raw/                  # Original datasets
│   └── processed/            # Processed data (when generated)
├── notebooks/                # Exploratory data analysis
├── src/
│   ├── models/               # Six forecasting model implementations
│   ├── data_pipeline.py      # Data preparation and splitting
│   ├── evaluate.py           # Model evaluation
│   └── visualize.py          # Evaluation visualizations
├── results/
│   ├── model_comparison/     # Validation and test metrics
│   ├── error_analysis/       # Validation and test error analysis
│   ├── per_seed/             # Neural Network seed results
│   ├── tuning/               # Hyperparameter tuning results
│   ├── sensitivity/          # Sensitivity experiments
│   ├── robustness/           # Robustness experiments
│   ├── ablation/             # Feature ablation experiments
│   └── README.md             # Detailed results documentation
├── figures/                  # Saved validation figures
├── main.py                   # Main model evaluation
├── error_analysis.py         # Validation error analysis
├── test_error_analysis.py    # Test error analysis
├── export_tuning.py          # Hyperparameter tuning exports
├── requirements.txt
└── README.md
```

## Running the Project

After completing the setup steps and activating the virtual environment, run the main model evaluation from the project root:

```bash
python main.py
```

This runs the six forecasting models, evaluates their performance on the validation and test datasets, and generates comparison results and visualizations.

To run the error analysis separately:

```bash
python error_analysis.py
```

This generates the validation error-analysis files in `results/error_analysis/`.

```bash
python test_error_analysis.py
```

This generates the test error-analysis files in `results/error_analysis/test/`.

To export the hyperparameter tuning results:

```bash
python export_tuning.py
```

These scripts may take time to complete because they perform model fitting and evaluation.

## Results and Documentation

The `results/` directory contains the model comparison metrics and supporting experimental outputs for Deliverable #5, including validation and test results, error analysis, tuning, sensitivity, robustness, and ablation experiments.

See [`results/README.md`](results/README.md) for detailed information about the datasets, result files, source scripts, and run references.
