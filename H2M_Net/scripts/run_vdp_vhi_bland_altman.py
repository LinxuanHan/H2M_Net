import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bland_altman import save_bland_altman
from src.dataset import MouseInferenceDataset
from src.diffusion import GaussianDiffusion
from src.model import SteeringSRUNet
from src.utils import ensure_dir, get_device, load_config, seed_everything, tensor_to_numpy
from src.vdp_vhi import compute_vdp_vhi


def parse_args():
    parser = argparse.ArgumentParser(description="Generate VDP/VHI Bland-Altman plots for Ours.")
    parser.add_argument("--config", default="configs/infer_mouse.yaml")
    parser.add_argument("--scales", nargs="+", type=int, default=[2, 4])
    parser.add_argument("--sampling_steps", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--omega", type=float, default=1.0)
    parser.add_argument("--noise_strength", type=float, default=0.1)
    parser.add_argument("--anchor_weight", type=float, default=0.2)
    parser.add_argument("--residual_scale", type=float, default=0.5)
    parser.add_argument("--gate_threshold", type=float, default=0.05)
    parser.add_argument("--output_dir", default="experiments/steered_diffusion/bland_altman")
    return parser.parse_args()


def load_model(cfg, checkpoint, device):
    model = SteeringSRUNet(**cfg["model"]).to(device)
    payload = torch.load(checkpoint, map_location=device)
    weight_key = "ema_model" if "ema_model" in payload else "model"
    model.load_state_dict(payload[weight_key])
    model.eval()
    return model


@torch.no_grad()
def sample_ours(diffusion, model, lr_up, scale, args, gate_config):
    gate_config = dict(gate_config or {})
    gate_config["threshold"] = args.gate_threshold
    return diffusion.ddim_steered_sample(
        model,
        lr_up,
        omega=args.omega,
        sampling_steps=args.sampling_steps,
        data_consistency_weight=1.0,
        init_mode="lr_noise",
        noise_strength=args.noise_strength,
        gate_config=gate_config,
        degradation_scale=scale,
        anchor_weight=args.anchor_weight,
        residual_scale=args.residual_scale,
        residual_clip=0.15,
        use_orthogonal=True,
        use_pathology_gate=True,
        use_residual_anchor=True,
    ).clamp(0.0, 1.0)


def write_metric_rows(rows, path):
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


def run_scale(cfg, scale, args, device):
    data_cfg = cfg["data"]
    dataset = MouseInferenceDataset(
        lr_dir=data_cfg[f"mouse_lr_x{scale}_dir"],
        hr_dir=data_cfg.get(f"mouse_hr_x{scale}_dir"),
        mask_dir=data_cfg.get("mouse_mask_dir"),
        scale=scale,
        limit=args.limit,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = load_model(cfg, cfg["inference"][f"checkpoint_x{scale}"], device)
    diffusion = GaussianDiffusion(**cfg["diffusion"]).to(device)
    gate_config = cfg["inference"].get("gate_config", None)
    vdp_cfg = cfg.get("vdp_vhi", {})
    metric_kwargs = {
        "mode": vdp_cfg.get("defect_threshold_mode", "relative_to_mean"),
        "relative_threshold": float(vdp_cfg.get("defect_relative_threshold", 0.6)),
        "percentile": float(vdp_cfg.get("defect_percentile", 15)),
        "fixed_threshold": float(vdp_cfg.get("fixed_threshold", 0.1)),
    }

    rows = []
    for batch in tqdm(loader, desc=f"VDP/VHI BA x{scale}"):
        lr_up = batch["lr_up"].to(device, non_blocking=True)
        sr = sample_ours(diffusion, model, lr_up, scale, args, gate_config)
        for index, name in enumerate(batch["name"]):
            if not bool(batch["has_hr"][index]):
                continue
            sr_np = tensor_to_numpy(sr[index])
            hr_np = tensor_to_numpy(batch["hr"][index])
            mask_np = tensor_to_numpy(batch["mask"][index]) if bool(batch["has_mask"][index]) else None
            hr_metrics = compute_vdp_vhi(hr_np, mask_np, **metric_kwargs)
            sr_metrics = compute_vdp_vhi(sr_np, mask_np, **metric_kwargs)
            rows.append(
                {
                    "name": name,
                    "scale": scale,
                    "hr_vdp": hr_metrics["vdp"],
                    "sr_vdp": sr_metrics["vdp"],
                    "vdp_error": sr_metrics["vdp"] - hr_metrics["vdp"],
                    "hr_vhi": hr_metrics["vhi"],
                    "sr_vhi": sr_metrics["vhi"],
                    "vhi_error": sr_metrics["vhi"] - hr_metrics["vhi"],
                }
            )

    out_dir = ensure_dir(Path(args.output_dir) / f"x{scale}")
    write_metric_rows(rows, out_dir / f"x{scale}_vdp_vhi_pairs.csv")
    hr_vdp = np.asarray([row["hr_vdp"] for row in rows], dtype=np.float32)
    sr_vdp = np.asarray([row["sr_vdp"] for row in rows], dtype=np.float32)
    hr_vhi = np.asarray([row["hr_vhi"] for row in rows], dtype=np.float32)
    sr_vhi = np.asarray([row["sr_vhi"] for row in rows], dtype=np.float32)
    save_bland_altman(
        hr_vdp,
        sr_vdp,
        f"x{scale} VDP Bland-Altman",
        out_dir / f"x{scale}_vdp_bland_altman.png",
        out_dir / f"x{scale}_vdp_bland_altman.csv",
    )
    save_bland_altman(
        hr_vhi,
        sr_vhi,
        f"x{scale} VHI Bland-Altman",
        out_dir / f"x{scale}_vhi_bland_altman.png",
        out_dir / f"x{scale}_vhi_bland_altman.csv",
    )
    return rows


def main():
    args = parse_args()
    seed_everything(42)
    cfg = load_config(args.config)
    device = get_device()
    all_rows = []
    for scale in args.scales:
        all_rows.extend(run_scale(cfg, scale, args, device))
    write_metric_rows(all_rows, Path(args.output_dir) / "all_scales_vdp_vhi_pairs.csv")
    print(f"Bland-Altman outputs saved to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
