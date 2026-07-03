from __future__ import annotations

from pathlib import Path
import csv

import numpy as np

from .utils import ensure_dir


def defect_mask(image: np.ndarray, mode: str = "relative_to_mean", relative_threshold: float = 0.6, percentile: float = 15, fixed_threshold: float = 0.1):
    image = np.squeeze(image).astype(np.float32)
    positive = image[image > 0]
    if positive.size == 0:
        return np.zeros_like(image, dtype=bool)
    if mode == "percentile":
        threshold = np.percentile(positive, percentile)
    elif mode == "fixed":
        threshold = fixed_threshold
    else:
        threshold = relative_threshold * float(positive.mean())
    return image < threshold


def compute_vdp_vhi(image: np.ndarray, mask: np.ndarray | None = None, **kwargs) -> dict[str, float]:
    image = np.squeeze(image).astype(np.float32)
    valid = np.squeeze(mask) > 0 if mask is not None and np.size(mask) else image > 0
    if not valid.any():
        valid = np.ones_like(image, dtype=bool)
    defect = defect_mask(image, **kwargs) & valid
    values = image[valid]
    vdp = 100.0 * float(defect.sum()) / float(valid.sum())
    vhi = float(values.std() / (values.mean() + 1e-8))
    return {"vdp": vdp, "vhi": vhi}


def save_vdp_vhi(rows: list[dict], path: str | Path):
    ensure_dir(Path(path).parent)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
