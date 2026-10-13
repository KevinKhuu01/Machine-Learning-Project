# Machine-Learning-Project
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
│   ├── raw/             # Original datasets
│   └── processed/       # Processed/model-ready data
├── notebooks/           # Exploratory data analysis notebooks
├── results/             # Model results and evaluation outputs
├── src/
│   ├── data_pipeline.py
│   ├── baseline_model.py
│   └── evaluation.py
├── main.py
├── requirements.txt
└── README.md
```