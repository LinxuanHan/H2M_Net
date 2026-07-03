from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import MouseInferenceDataset
from src.h2m_net import H2MSteeredDiffusion
from src.metrics import compute_metrics, save_metric_tables
from src.model import SteeringSRUNet
from src.utils import ensure_dir, load_config, save_png, seed_everything, tensor_to_numpy


def parse_args():
    parser = argparse.ArgumentParser(description="H2M-Net target-free zero-shot mouse SR inference.")
    parser.add_argument("--config", default="configs/h2m_infer_mouse.yaml")
    parser.add_argument("--scale", type=int, default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--sampling_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--omega", type=float, default=None)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--noise_strength", type=float, default=None)
    parser.add_argument("--init_mode", choices=["lr_noise", "random"], default=None)
    parser.add_argument("--disable_lra", action="store_true")
    parser.add_argument("--use_raw_weights", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(int(cfg.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    infer_cfg = cfg["inference"]
    data_cfg = cfg["data"]
    scale = int(args.scale or infer_cfg.get("scale", 2))
    checkpoint = args.checkpoint or infer_cfg[f"checkpoint_x{scale}"]
    sampling_steps = int(args.sampling_steps or infer_cfg.get("sampling_steps", 50))
    omega = float(args.omega if args.omega is not None else infer_cfg.get("omega", 1.0))
    tau = float(args.tau if args.tau is not None else infer_cfg.get("tau", 0.1))
    gamma = float(args.gamma if args.gamma is not None else infer_cfg.get("gamma", 0.02))
    noise_strength = float(args.noise_strength if args.noise_strength is not None else infer_cfg.get("noise_strength", 0.2))
    init_mode = args.init_mode or infer_cfg.get("init_mode", "lr_noise")

    dataset = MouseInferenceDataset(
        lr_dir=data_cfg[f"mouse_lr_x{scale}_dir"],
        hr_dir=data_cfg.get(f"mouse_hr_x{scale}_dir"),
        mask_dir=data_cfg.get("mouse_mask_dir"),
        scale=scale,
        limit=args.limit,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = SteeringSRUNet(**cfg["model"]).to(device)
    payload = torch.load(checkpoint, map_location=device)
    weight_key = "model" if args.use_raw_weights or "ema_model" not in payload else "ema_model"
    model.load_state_dict(payload[weight_key])
    model.eval()

    h2m = H2MSteeredDiffusion(
        model,
        scale=scale,
        timesteps=int(cfg["diffusion"].get("timesteps", 1000)),
        degradation_mode=cfg["diffusion"].get("degradation_mode", "bicubic"),
    ).to(device)
    h2m.eval()

    tag = f"x{scale}_omega{omega:g}_tau{tau:g}_gamma{gamma:g}_ns{noise_strength:g}"
    sr_dir = ensure_dir(Path(cfg["output"]["sr_dir"]) / tag)
    metric_dir = ensure_dir(cfg["output"]["metric_dir"])
    rows = []
    for batch in tqdm(loader, desc=f"H2M inference x{scale}"):
        lr_up = batch["lr_up"].to(device, non_blocking=True)
        sr = h2m.sample(
            lr_up,
            sampling_steps=sampling_steps,
            noise_strength=noise_strength,
            omega=omega,
            tau=tau,
            gamma=gamma,
            init_mode=init_mode,
            apply_lra=not args.disable_lra,
        )
        for index, name in enumerate(batch["name"]):
            sr_np = tensor_to_numpy(sr[index])
            save_png(sr_np, sr_dir / f"{name}.png")
            if bool(batch["has_hr"][index]):
                hr_np = tensor_to_numpy(batch["hr"][index])
                mask_np = tensor_to_numpy(batch["mask"][index]) if bool(batch["has_mask"][index]) else None
                row = {"name": name, "scale": scale, "omega": omega, "tau": tau, "gamma": gamma}
                row.update(compute_metrics(sr_np, hr_np, mask_np if cfg.get("evaluation", {}).get("use_mask", True) else None))
                rows.append(row)
    if rows:
        summary = save_metric_tables(
            rows,
            metric_dir / f"h2m_{tag}_per_image.csv",
            metric_dir / f"h2m_{tag}_summary.csv",
        )
        print(summary)
    print(f"H2M inference done. results={sr_dir.resolve()}")


if __name__ == "__main__":
    main()
