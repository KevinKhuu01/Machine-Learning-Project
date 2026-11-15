import os
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = "figures"


def _flat(a):
    """Convert Series / DataFrame / 2-D array to a flat 1-D NumPy array."""
    return np.asarray(a).ravel()


def _save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUT_DIR, name), dpi=150, bbox_inches="tight")


def plot_metric_comparison(results):
    names = list(results.keys())
    mae = [results[n][0] for n in names]
    rmse = [results[n][1] for n in names]
    x = np.arange(len(names))
    w = 0.38

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w / 2, mae, w, label="MAE")
    ax.bar(x + w / 2, rmse, w, label="RMSE")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Error (vehicles per interval)")
    ax.set_title("Model comparison on validation set")
    ax.legend()
    _save(fig, "metric_comparison.png")


def plot_pred_vs_actual(y_valid, preds):
    y = _flat(y_valid)
    n = len(preds)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
    axes = np.atleast_1d(axes).ravel()
    lims = [y.min(), y.max()]

    for ax, (name, p) in zip(axes, preds.items()):
        ax.scatter(y, _flat(p), s=6, alpha=0.4)
        ax.plot(lims, lims, "r--", lw=1)  # perfect-prediction line
        ax.set_title(name)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle("Predicted vs actual")
    fig.tight_layout()
    _save(fig, "pred_vs_actual.png")


def plot_time_series(y_valid, preds, n_points=500, models=None):
    models = models or ["naive_benchmark", "decision_tree", "neural_network"]
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(_flat(y_valid)[:n_points], color="black", lw=2, label="Actual")
    for m in models:
        ax.plot(_flat(preds[m])[:n_points], lw=1, alpha=0.8, label=m)
    ax.set_xlabel("Interval (validation set)")
    ax.set_ylabel("Vehicle volume")
    ax.set_title(f"First {n_points} validation intervals")
    ax.legend()
    _save(fig, "time_series.png")


def plot_residuals(y_valid, preds):
    y = _flat(y_valid)
    names = list(preds.keys())
    resid = [y - _flat(preds[n]) for n in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(resid, showfliers=False)
    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.axhline(0, color="red", ls="--", lw=1)
    ax.set_ylabel("Residual (actual - predicted)")
    ax.set_title("Residual distribution by model")
    _save(fig, "residuals.png")


def plot_all(y_valid, results, preds):
    # Sanity check: every prediction should match the length of y_valid
    n = len(_flat(y_valid))
    for name, p in preds.items():
        if len(_flat(p)) != n:
            raise ValueError(f"{name}: got {len(_flat(p))} predictions, expected {n}")

    plot_metric_comparison(results)
    plot_pred_vs_actual(y_valid, preds)
    plot_time_series(y_valid, preds)
    plot_residuals(y_valid, preds)
    plt.show()