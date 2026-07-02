import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


DATASETS = {
    "human_val": ("human", "val"),
    "mouse_test": ("mouse", "test"),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run bicubic super-resolution baseline on 129Xe lung MRI slices."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-root", type=Path, default=Path("experiments") / "bicubic")
    parser.add_argument("--scales", type=int, nargs="+", default=[2, 4], choices=[2, 4])
    parser.add_argument(
        "--sets",
        nargs="+",
        default=["human_val", "mouse_test"],
        choices=sorted(DATASETS.keys()),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_grayscale(path):
    return Image.open(path).convert("L")


def image_to_array(image):
    return np.asarray(image, dtype=np.float32)


def compute_metrics(sr_image, hr_image):
    sr = image_to_array(sr_image)
    hr = image_to_array(hr_image)
    diff = sr - hr
    return {
        "psnr": float(peak_signal_noise_ratio(hr, sr, data_range=255)),
        "ssim": float(structural_similarity(hr, sr, data_range=255)),
        "mse": float(np.mean(diff * diff)),
        "mae": float(np.mean(np.abs(diff))),
    }


def resize_bicubic(lr_image, target_size):
    return lr_image.resize(target_size, Image.Resampling.BICUBIC)


def collect_pairs(data_root, set_name, scale):
    species, split = DATASETS[set_name]
    split_root = data_root / species / split
    raw_dir = split_root / "raw"
    lr_dir = split_root / f"down_{scale}x_bilinear"

    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Missing HR directory: {raw_dir}")
    if not lr_dir.is_dir():
        raise FileNotFoundError(f"Missing LR directory: {lr_dir}")

    pairs = []
    for hr_path in sorted(raw_dir.glob("*.png")):
        lr_path = lr_dir / hr_path.name
        if not lr_path.is_file():
            raise FileNotFoundError(f"Missing LR pair for {hr_path}: {lr_path}")
        pairs.append((lr_path, hr_path))

    if not pairs:
        raise RuntimeError(f"No PNG pairs found in {split_root}")
    return pairs


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "set",
        "scale",
        "filename",
        "width",
        "height",
        "psnr",
        "ssim",
        "mse",
        "mae",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    summary = {
        "count": len(rows),
        "psnr_mean": float(np.mean([row["psnr"] for row in rows])),
        "psnr_std": float(np.std([row["psnr"] for row in rows])),
        "ssim_mean": float(np.mean([row["ssim"] for row in rows])),
        "ssim_std": float(np.std([row["ssim"] for row in rows])),
        "mse_mean": float(np.mean([row["mse"] for row in rows])),
        "mse_std": float(np.std([row["mse"] for row in rows])),
        "mae_mean": float(np.mean([row["mae"] for row in rows])),
        "mae_std": float(np.std([row["mae"] for row in rows])),
    }
    return summary


def run_set(data_root, output_root, set_name, scale, overwrite):
    pairs = collect_pairs(data_root, set_name, scale)
    result_root = output_root / f"x{scale}" / set_name
    sr_dir = result_root / "sr"
    metrics_dir = result_root / "metrics"
    sr_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for lr_path, hr_path in pairs:
        hr_image = load_grayscale(hr_path)
        lr_image = load_grayscale(lr_path)
        sr_image = resize_bicubic(lr_image, hr_image.size)

        sr_path = sr_dir / hr_path.name
        if overwrite or not sr_path.exists():
            sr_image.save(sr_path)

        metrics = compute_metrics(sr_image, hr_image)
        rows.append(
            {
                "set": set_name,
                "scale": scale,
                "filename": hr_path.name,
                "width": hr_image.width,
                "height": hr_image.height,
                **metrics,
            }
        )

    summary = summarize(rows)
    write_csv(metrics_dir / "per_image_metrics.csv", rows)
    with (metrics_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    return summary


def main():
    args = parse_args()
    all_summaries = {}

    for scale in args.scales:
        for set_name in args.sets:
            summary = run_set(
                args.data_root,
                args.output_root,
                set_name,
                scale,
                args.overwrite,
            )
            all_summaries[f"x{scale}/{set_name}"] = summary
            print(
                f"x{scale} {set_name}: "
                f"n={summary['count']}, "
                f"PSNR={summary['psnr_mean']:.4f}, "
                f"SSIM={summary['ssim_mean']:.4f}, "
                f"MSE={summary['mse_mean']:.4f}, "
                f"MAE={summary['mae_mean']:.4f}"
            )

    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / "summary_all.json").open("w", encoding="utf-8") as file:
        json.dump(all_summaries, file, indent=2)


if __name__ == "__main__":
    main()
