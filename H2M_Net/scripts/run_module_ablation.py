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
    parser = argparse.ArgumentParser(description="Run main-paper module ablation.")
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
    parser.add_argument("--output_dir", default="experiments/steered_diffusion/module_ablation")
    return parser.parse_args()


def build_variants():
    return [
        {"name": "Base", "lr_anchor": False, "ors": False, "psg": False},
        {"name": "Base+LR Anchor", "lr_anchor": True, "ors": False, "psg": False},
        {"name": "Base+ORS", "lr_anchor": False, "ors": True, "psg": False},
        {"name": "Base+PSG", "lr_anchor": False, "ors": False, "psg": True},
        {"name": "w/o LR Anchor", "lr_anchor": False, "ors": True, "psg": True},
        {"name": "w/o ORS", "lr_anchor": True, "ors": False, "psg": True},
        {"name": "w/o PSG", "lr_anchor": True, "ors": True, "psg": False},
        {"name": "Full", "lr_anchor": True, "ors": True, "psg": True},
    ]


def load_model(cfg, checkpoint, device):
    model = SteeringSRUNet(**cfg["model"]).to(device)
    payload = torch.load(checkpoint, map_location=device)
    weight_key = "ema_model" if "ema_model" in payload else "model"
    model.load_state_dict(payload[weight_key])
    model.eval()
    return model


@torch.no_grad()
def sample_batch(diffusion, model, lr_up, scale, variant, args, gate_config):
    gate_config = dict(gate_config or {})
    gate_config["threshold"] = args.gate_threshold
    lr_anchor = bool(variant["lr_anchor"])
    return diffusion.ddim_steered_sample(
        model,
        lr_up,
        omega=args.omega,
        sampling_steps=args.sampling_steps,
        data_consistency_weight=1.0 if lr_anchor else 0.0,
        init_mode="lr_noise" if lr_anchor else "random",
        noise_strength=args.noise_strength,
        gate_config=gate_config,
        degradation_scale=scale,
        anchor_weight=args.anchor_weight if lr_anchor else 0.0,
        residual_scale=args.residual_scale,
        residual_clip=0.15,
        use_orthogonal=bool(variant["ors"]),
        use_pathology_gate=bool(variant["psg"]),
        use_residual_anchor=lr_anchor,
    )


def run_variant(cfg, scale, variant, args, device):
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

    rows = []
    for batch in tqdm(loader, desc=f"x{scale} {variant['name']}"):
        names = batch["name"]
        lr_up = batch["lr_up"].to(device, non_blocking=True)
        sr = sample_batch(diffusion, model, lr_up, scale, variant, args, gate_config)
        for index, name in enumerate(names):
            if not bool(batch["has_hr"][index]):
                continue
            row = {
                "name": name,
                "variant": variant["name"],
                "scale": scale,
                "lr_anchor": variant["lr_anchor"],
                "ors": variant["ors"],
                "psg": variant["psg"],
                "omega": args.omega,
                "noise_strength": args.noise_strength,
                "anchor_weight": args.anchor_weight,
                "residual_scale": args.residual_scale,
                "gate_threshold": args.gate_threshold,
            }
            row.update(compute_metrics(tensor_to_numpy(sr[index]), tensor_to_numpy(batch["hr"][index])))
            rows.append(row)

    safe_name = variant["name"].replace("/", "wo").replace("+", "_plus_").replace(" ", "_")
    out_dir = ensure_dir(Path(args.output_dir) / f"x{scale}")
    summary = save_metric_tables(
        rows,
        out_dir / f"{safe_name}_per_image.csv",
        out_dir / f"{safe_name}_summary.csv",
    )
    return {"variant": variant["name"], "scale": scale, **summary}


def write_master_table(rows, output_dir):
    output_dir = ensure_dir(output_dir)
    variants = [variant["name"] for variant in build_variants()]
    by_variant = {name: {"variant": name} for name in variants}
    for row in rows:
        prefix = f"x{row['scale']}"
        item = by_variant[row["variant"]]
        item[f"{prefix}_psnr"] = row["psnr_mean"]
        item[f"{prefix}_ssim"] = row["ssim_mean"]
        item[f"{prefix}_mae"] = row["mae_mean"]
        item[f"{prefix}_rmse"] = row["rmse_mean"]

    csv_path = output_dir / "module_ablation_master_table.csv"
    fieldnames = ["variant", "x2_psnr", "x2_ssim", "x4_psnr", "x4_ssim", "x2_mae", "x2_rmse", "x4_mae", "x4_rmse"]
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for name in variants:
            writer.writerow(by_variant[name])

    md_path = output_dir / "module_ablation_paper_table.md"
    with open(md_path, "w", encoding="utf-8") as file:
        file.write("| Variant | LR Anchor | ORS | PSG | x2 PSNR | x2 SSIM | x4 PSNR | x4 SSIM |\n")
        file.write("|---|:---:|:---:|:---:|---:|---:|---:|---:|\n")
        variant_cfg = {variant["name"]: variant for variant in build_variants()}
        for name in variants:
            item = by_variant[name]
            cfg = variant_cfg[name]
            file.write(
                f"| {name} | {'✓' if cfg['lr_anchor'] else '✗'} | {'✓' if cfg['ors'] else '✗'} | {'✓' if cfg['psg'] else '✗'} | "
                f"{float(item.get('x2_psnr', 0)):.4f} | {float(item.get('x2_ssim', 0)):.4f} | "
                f"{float(item.get('x4_psnr', 0)):.4f} | {float(item.get('x4_ssim', 0)):.4f} |\n"
            )
    return csv_path, md_path


def main():
    args = parse_args()
    seed_everything(42)
    cfg = load_config(args.config)
    device = get_device()
    rows = []
    for scale in args.scales:
        for variant in build_variants():
            rows.append(run_variant(cfg, scale, variant, args, device))
    csv_path, md_path = write_master_table(rows, Path(args.output_dir))
    print(f"Module ablation done: {csv_path}")
    print(f"Paper table: {md_path}")


if __name__ == "__main__":
    main()
