from __future__ import annotations

from pathlib import Path
import csv

import numpy as np

from .utils import ensure_dir


def _masked(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None):
    a = np.squeeze(a).astype(np.float32)
    b = np.squeeze(b).astype(np.float32)
    if mask is None:
        return a, b
    mask = np.squeeze(mask) > 0
    if not mask.any():
        return a, b
    return a[mask], b[mask]


def _psnr(reference: np.ndarray, prediction: np.ndarray, data_range: float = 1.0) -> float:
    mse = float(np.mean((reference.astype(np.float32) - prediction.astype(np.float32)) ** 2))
    if mse <= 1e-12:
        return float("inf")
    return float(20.0 * np.log10(data_range) - 10.0 * np.log10(mse))


def _uniform_filter_2d(image: np.ndarray, window: int) -> np.ndarray:
    pad = window // 2
    padded = np.pad(image.astype(np.float32), ((pad, pad), (pad, pad)), mode="reflect")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    return (
        integral[window:, window:]
        - integral[:-window, window:]
        - integral[window:, :-window]
        + integral[:-window, :-window]
    ) / float(window * window)


def _ssim(reference: np.ndarray, prediction: np.ndarray, data_range: float = 1.0, window: int = 7) -> float:
    reference = reference.astype(np.float32)
    prediction = prediction.astype(np.float32)
    min_side = min(reference.shape[-2:])
    window = min(window, min_side if min_side % 2 == 1 else min_side - 1)
    window = max(3, window)
    mu_x = _uniform_filter_2d(reference, window)
    mu_y = _uniform_filter_2d(prediction, window)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y
    sigma_x2 = _uniform_filter_2d(reference * reference, window) - mu_x2
    sigma_y2 = _uniform_filter_2d(prediction * prediction, window) - mu_y2
    sigma_xy = _uniform_filter_2d(reference * prediction, window) - mu_xy
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / ((mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2) + 1e-12)
    return float(np.mean(ssim_map))


def compute_metrics(sr: np.ndarray, hr: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    sr_img = np.squeeze(sr).astype(np.float32)
    hr_img = np.squeeze(hr).astype(np.float32)
    sr_vec, hr_vec = _masked(sr_img, hr_img, mask)
    mae = float(np.mean(np.abs(sr_vec - hr_vec)))
    rmse = float(np.sqrt(np.mean((sr_vec - hr_vec) ** 2)))
    psnr = _psnr(hr_img, sr_img, data_range=1.0)
    ssim = _ssim(hr_img, sr_img, data_range=1.0, window=7)
    return {"psnr": psnr, "ssim": ssim, "mae": mae, "rmse": rmse}


def save_metric_tables(rows: list[dict], csv_path: str | Path, summary_path: str | Path) -> dict[str, float]:
    ensure_dir(Path(csv_path).parent)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    metric_cols = [col for col in ["psnr", "ssim", "mae", "rmse"] if any(col in row for row in rows)]
    summary = {}
    for col in metric_cols:
        values = np.asarray([float(row[col]) for row in rows if col in row], dtype=np.float32)
        summary[f"{col}_mean"] = float(values.mean())
        summary[f"{col}_std"] = float(values.std())
    with open(summary_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=sorted(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    return summary
