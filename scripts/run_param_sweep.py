import argparse
import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import MouseInferenceDataset
from src.diffusion import GaussianDiffusion
from src.metrics import compute_metrics, save_metric_tables
from src.model import SteeringSRUNet
from src.utils import ensure_dir, get_device, load_config, seed_everything, tensor_to_numpy


def parse_args():
    parser = argparse.ArgumentParser(description="Run one-factor parameter sweeps for steered diffusion SR.")
    parser.add_argument("--config", default="configs/infer_mouse.yaml")
    parser.add_argument("--scales", nargs="+", type=int, default=[2, 4])
    parser.add_argument("--sampling_steps", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output_dir", default="experiments/steered_diffusion/param_sweep")
    return parser.parse_args()


BASE_PARAMS = {
    "omega": 1.0,
    "noise_strength": 0.2,
    "anchor_weight": 0.4,
    "residual_scale": 0.5,
    "bicubic_blend": 0.0,
    "gate_threshold": 0.15,
    "use_orthogonal": True,
    "use_pathology_gate": True,
}


SWEEPS = {
    "omega": [0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0],
    "noise_strength": [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5],
    "anchor_weight": [0.0, 0.2, 0.4, 0.6, 0.8],
    "residual_scale": [0.3, 0.5, 0.7, 0.9, 1.1],
    "bicubic_blend": [0.0, 0.05, 0.1, 0.15, 0.25, 0.4],
    "gate_threshold": [0.05, 0.1, 0.15, 0.2, 0.3],
}


def build_variants():
    variants = []
    for parameter, values in SWEEPS.items():
        for value in values:
            params = dict(BASE_PARAMS)
            params[parameter] = value
            params["sweep"] = parameter
            params["value"] = value
            params["name"] = f"{parameter}={value:g}"
            variants.append(params)
    return variants


def load_model(cfg, checkpoint, device):
    model = SteeringSRUNet(**cfg["model"]).to(device)
    payload = torch.load(checkpoint, map_location=device)
    weight_key = "ema_model" if "ema_model" in payload else "model"
    model.load_state_dict(payload[weight_key])
    model.eval()
    return model


@torch.no_grad()
def sample_batch(diffusion, model, lr_up, scale, params, sampling_steps, gate_config):
    gate_config = dict(gate_config or {})
    gate_config["threshold"] = params["gate_threshold"]
    sr = diffusion.ddim_steered_sample(
        model,
        lr_up,
        omega=params["omega"],
        sampling_steps=sampling_steps,
        data_consistency_weight=1.0,
        init_mode="lr_noise",
        noise_strength=params["noise_strength"],
        gate_config=gate_config,
        degradation_scale=scale,
        anchor_weight=params["anchor_weight"],
        residual_scale=params["residual_scale"],
        residual_clip=0.15,
        use_orthogonal=params["use_orthogonal"],
        use_pathology_gate=params["use_pathology_gate"],
    )
    if params["bicubic_blend"] > 0:
        sr = ((1.0 - params["bicubic_blend"]) * sr + params["bicubic_blend"] * lr_up).clamp(0.0, 1.0)
    return sr.clamp(0.0, 1.0)


def run_variant(cfg, scale, params, args, device, dataset, loader):
    checkpoint = cfg["inference"][f"checkpoint_x{scale}"]
    model = load_model(cfg, checkpoint, device)
    diffusion = GaussianDiffusion(**cfg["diffusion"]).to(device)
    gate_config = cfg["inference"].get("gate_config", None)

    rows = []
    progress = tqdm(loader, desc=f"x{scale} {params['name']}")
    for batch in progress:
        names = batch["name"]
        lr_up = batch["lr_up"].to(device, non_blocking=True)
        sr = sample_batch(diffusion, model, lr_up, scale, params, args.sampling_steps, gate_config)
        for index, name in enumerate(names):
            if not bool(batch["has_hr"][index]):
                continue
            sr_np = tensor_to_numpy(sr[index])
            hr_np = tensor_to_numpy(batch["hr"][index])
            row = {
                "name": name,
                "scale": scale,
                "sweep": params["sweep"],
                "value": params["value"],
                "variant": params["name"],
                "omega": params["omega"],
                "noise_strength": params["noise_strength"],
                "anchor_weight": params["anchor_weight"],
                "residual_scale": params["residual_scale"],
                "bicubic_blend": params["bicubic_blend"],
                "gate_threshold": params["gate_threshold"],
            }
            row.update(compute_metrics(sr_np, hr_np))
            rows.append(row)

    safe_name = params["name"].replace("=", "_").replace(".", "p")
    out_dir = ensure_dir(Path(args.output_dir) / f"x{scale}" / params["sweep"])
    summary = save_metric_tables(
        rows,
        out_dir / f"{safe_name}_per_image.csv",
        out_dir / f"{safe_name}_summary.csv",
    )
    return {
        "scale": scale,
        "sweep": params["sweep"],
        "value": params["value"],
        "variant": params["name"],
        **summary,
    }


def write_tables(rows, output_dir):
    output_dir = ensure_dir(output_dir)
    all_path = output_dir / "param_sweep_all.csv"
    fieldnames = ["scale", "sweep", "value", "variant", "psnr_mean", "ssim_mean", "mae_mean", "rmse_mean"]
    with open(all_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    best_rows = []
    groups = sorted({(row["scale"], row["sweep"]) for row in rows})
    for scale, sweep in groups:
        candidates = [row for row in rows if row["scale"] == scale and row["sweep"] == sweep]
        best = max(candidates, key=lambda item: float(item["psnr_mean"]))
        best_rows.append(best)
    best_path = output_dir / "param_sweep_best_by_psnr.csv"
    with open(best_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in best_rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    md_path = output_dir / "param_sweep_best_by_psnr.md"
    with open(md_path, "w", encoding="utf-8") as file:
        file.write("| Scale | Parameter | Best value | PSNR ↑ | SSIM ↑ |\n")
        file.write("|---:|---|---:|---:|---:|\n")
        for row in best_rows:
            file.write(
                f"| x{row['scale']} | {row['sweep']} | {float(row['value']):g} | "
                f"{float(row['psnr_mean']):.4f} | {float(row['ssim_mean']):.4f} |\n"
            )
    return all_path, best_path, md_path


def main():
    args = parse_args()
    seed_everything(42)
    cfg = load_config(args.config)
    device = get_device()
    rows = []
    variants = build_variants()
    for scale in args.scales:
        data_cfg = cfg["data"]
        dataset = MouseInferenceDataset(
            lr_dir=data_cfg[f"mouse_lr_x{scale}_dir"],
            hr_dir=data_cfg.get(f"mouse_hr_x{scale}_dir"),
            mask_dir=data_cfg.get("mouse_mask_dir"),
            scale=scale,
            limit=args.limit,
        )
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        for params in variants:
            rows.append(run_variant(cfg, scale, params, args, device, dataset, loader))
    all_path, best_path, md_path = write_tables(rows, Path(args.output_dir))
    print(f"Parameter sweep done: {all_path}")
    print(f"Best-by-PSNR table: {best_path}")
    print(f"Markdown table: {md_path}")


if __name__ == "__main__":
    main()
