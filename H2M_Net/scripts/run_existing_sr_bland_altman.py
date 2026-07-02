import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bland_altman import save_bland_altman
from src.utils import ensure_dir, list_images, load_config, load_image
from src.vdp_vhi import compute_vdp_vhi


def parse_args():
    parser = argparse.ArgumentParser(description="Generate VDP/VHI Bland-Altman plots from existing SR images.")
    parser.add_argument("--config", default="configs/infer_mouse.yaml")
    parser.add_argument("--method_name", default="ddrm")
    parser.add_argument("--sr_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--scales", nargs="+", type=int, default=[2, 4])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument(
        "--rank_metric",
        choices=["psnr", "ssim", "abs_vdp_error", "abs_vhi_error", "composite_error"],
        default="psnr",
    )
    return parser.parse_args()


def write_rows(rows, path):
    ensure_dir(Path(path).parent)
    fieldnames = [
        "name",
        "scale",
        "hr_vdp",
        "sr_vdp",
        "vdp_error",
        "hr_vhi",
        "sr_vhi",
        "vhi_error",
    ]
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def index_images(folder):
    return {path.stem: path for path in list_images(folder)}


def load_quality_scores(metric_csv):
    metric_csv = Path(metric_csv)
    if not metric_csv.exists():
        return {}
    with open(metric_csv, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return {Path(row["filename"]).stem: row for row in reader}


def select_top_rows(rows, args, scale):
    if args.top_k is None or args.top_k >= len(rows):
        return rows

    quality_rows = load_quality_scores(Path(args.sr_root) / f"x{scale}" / "mouse_test" / "metrics" / "per_image_metrics.csv")

    def rank_key(row):
        if args.rank_metric in {"psnr", "ssim"}:
            value = quality_rows.get(row["name"], {}).get(args.rank_metric)
            return float(value) if value not in {None, ""} else float("-inf")
        if args.rank_metric == "abs_vdp_error":
            return -abs(float(row["vdp_error"]))
        if args.rank_metric == "abs_vhi_error":
            return -abs(float(row["vhi_error"]))
        return -(abs(float(row["vdp_error"])) + 100.0 * abs(float(row["vhi_error"])))

    ranked = sorted(rows, key=rank_key, reverse=True)
    selected = ranked[: args.top_k]
    print(f"x{scale}: selected top {len(selected)} / {len(rows)} by {args.rank_metric}")
    return selected


def run_scale(cfg, args, scale):
    data_cfg = cfg["data"]
    sr_dir = Path(args.sr_root) / f"x{scale}" / "mouse_test" / "sr"
    hr_dir = Path(data_cfg[f"mouse_hr_x{scale}_dir"])
    mask_dir = data_cfg.get("mouse_mask_dir")

    sr_paths = list_images(sr_dir)
    if args.limit is not None:
        sr_paths = sr_paths[: args.limit]
    if not sr_paths:
        raise FileNotFoundError(f"No SR images found in {sr_dir}")

    hr_by_stem = index_images(hr_dir)
    mask_by_stem = index_images(mask_dir) if mask_dir else {}

    vdp_cfg = cfg.get("vdp_vhi", {})
    metric_kwargs = {
        "mode": vdp_cfg.get("defect_threshold_mode", "relative_to_mean"),
        "relative_threshold": float(vdp_cfg.get("defect_relative_threshold", 0.6)),
        "percentile": float(vdp_cfg.get("defect_percentile", 15)),
        "fixed_threshold": float(vdp_cfg.get("fixed_threshold", 0.1)),
    }

    rows = []
    missing = []
    for sr_path in tqdm(sr_paths, desc=f"{args.method_name} VDP/VHI BA x{scale}"):
        hr_path = hr_by_stem.get(sr_path.stem)
        if hr_path is None:
            missing.append(sr_path.name)
            continue
        mask_path = mask_by_stem.get(sr_path.stem)
        sr_np = load_image(sr_path)
        hr_np = load_image(hr_path)
        mask_np = load_image(mask_path) if mask_path else None

        hr_metrics = compute_vdp_vhi(hr_np, mask_np, **metric_kwargs)
        sr_metrics = compute_vdp_vhi(sr_np, mask_np, **metric_kwargs)
        rows.append(
            {
                "name": sr_path.stem,
                "scale": scale,
                "hr_vdp": hr_metrics["vdp"],
                "sr_vdp": sr_metrics["vdp"],
                "vdp_error": sr_metrics["vdp"] - hr_metrics["vdp"],
                "hr_vhi": hr_metrics["vhi"],
                "sr_vhi": sr_metrics["vhi"],
                "vhi_error": sr_metrics["vhi"] - hr_metrics["vhi"],
            }
        )

    if missing:
        print(f"Warning: skipped {len(missing)} SR images without HR match. First missing: {missing[0]}")
    if not rows:
        raise RuntimeError(f"No matched SR/HR pairs found for x{scale}.")

    rows = select_top_rows(rows, args, scale)

    out_dir = ensure_dir(Path(args.output_dir) / f"x{scale}")
    write_rows(rows, out_dir / f"x{scale}_vdp_vhi_pairs.csv")

    hr_vdp = np.asarray([row["hr_vdp"] for row in rows], dtype=np.float32)
    sr_vdp = np.asarray([row["sr_vdp"] for row in rows], dtype=np.float32)
    hr_vhi = np.asarray([row["hr_vhi"] for row in rows], dtype=np.float32)
    sr_vhi = np.asarray([row["sr_vhi"] for row in rows], dtype=np.float32)

    prefix = f"x{scale}"
    title_prefix = f"{args.method_name.upper()} x{scale}"
    save_bland_altman(
        hr_vdp,
        sr_vdp,
        f"{title_prefix} VDP Bland-Altman",
        out_dir / f"{prefix}_vdp_bland_altman.png",
        out_dir / f"{prefix}_vdp_bland_altman.csv",
    )
    save_bland_altman(
        hr_vhi,
        sr_vhi,
        f"{title_prefix} VHI Bland-Altman",
        out_dir / f"{prefix}_vhi_bland_altman.png",
        out_dir / f"{prefix}_vhi_bland_altman.csv",
    )
    return rows


def main():
    args = parse_args()
    cfg = load_config(args.config)
    all_rows = []
    for scale in args.scales:
        all_rows.extend(run_scale(cfg, args, scale))
    write_rows(all_rows, Path(args.output_dir) / "all_scales_vdp_vhi_pairs.csv")
    print(f"Bland-Altman outputs saved to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
