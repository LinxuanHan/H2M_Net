from __future__ import annotations

from pathlib import Path
import csv

import numpy as np

from .utils import ensure_dir


def save_bland_altman(reference, prediction, title: str, figure_path: str | Path, csv_path: str | Path):
    reference = np.asarray(reference, dtype=np.float32)
    prediction = np.asarray(prediction, dtype=np.float32)
    mean = (reference + prediction) / 2
    diff = prediction - reference
    bias = float(diff.mean())
    loa = 1.96 * float(diff.std(ddof=0))
    ensure_dir(Path(csv_path).parent)
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["mean", "diff", "bias", "loa95"])
        writer.writeheader()
        for mean_value, diff_value in zip(mean, diff):
            writer.writerow({"mean": float(mean_value), "diff": float(diff_value), "bias": bias, "loa95": loa})
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    ensure_dir(Path(figure_path).parent)
    plt.figure(figsize=(5, 4))
    plt.scatter(mean, diff, s=12, alpha=0.7)
    plt.axhline(bias, color="red", label=f"bias={bias:.4f}")
    plt.axhline(bias + loa, color="gray", linestyle="--")
    plt.axhline(bias - loa, color="gray", linestyle="--")
    plt.xlabel("Mean")
    plt.ylabel("Prediction - Reference")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path, dpi=200)
    plt.close()
