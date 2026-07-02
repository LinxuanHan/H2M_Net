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
    parser = argparse.ArgumentParser(description="Run inference-time ablations for steered diffusion SR.")
    parser.add_argument("--config", default="configs/infer_mouse.yaml")
    parser.add_argument("--scales", nargs="+", type=int, default=[2, 4])
    parser.add_argument("--sampling_steps", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--include_self_ensemble", action="store_true")
    parser.add_argument("--output_dir", default="experiments/steered_diffusion/ablation")
    return parser.parse_args()


def build_variants(include_self_ensemble=False):
    variants = [
        {
            "name": "Full",
            "omega": 3.0,
            "noise_strength": 0.2,
            "anchor_weight": 0.4,
            "residual_scale": 0.5,
            "bicubic_blend": 0.0,
            "use_orthogonal": True,
            "use_pathology_gate": True,
            "self_ensemble": False,
        },
        {
            "name": "w/o steering",
            "omega": 0.0,
            "noise_strength": 0.2,
            "anchor_weight": 0.4,
            "residual_scale": 0.5,
            "bicubic_blend": 0.0,
            "use_orthogonal": True,
            "use_pathology_gate": True,
            "self_ensemble": False,
        },
        {
            "name": "weak steering",
            "omega": 1.0,
            "noise_strength": 0.2,
            "anchor_weight": 0.4,
            "residual_scale": 0.5,
            "bicubic_blend": 0.0,
            "use_orthogonal": True,
            "use_pathology_gate": True,
            "self_ensemble": False,
        },
        {
            "name": "strong steering",
            "omega": 4.0,
            "noise_strength": 0.2,
            "anchor_weight": 0.4,
            "residual_scale": 0.5,
            "bicubic_blend": 0.0,
            "use_orthogonal": True,
            "use_pathology_gate": True,
            "self_ensemble": False,
        },
        {
            "name": "w/o orthogonal projection",
            "omega": 3.0,
            "noise_strength": 0.2,
            "anchor_weight": 0.4,
            "residual_scale": 0.5,
            "bicubic_blend": 0.0,
            "use_orthogonal": False,
            "use_pathology_gate": True,
            "self_ensemble": False,
        },
        {
            "name": "w/o pathology gate",
            "omega": 3.0,
            "noise_strength": 0.2,
            "anchor_weight": 0.4,
            "residual_scale": 0.5,
            "bicubic_blend": 0.0,
            "use_orthogonal": True,
            "use_pathology_gate": False,
            "self_ensemble": False,
        },
        {
            "name": "w/o anchor",
            "omega": 3.0,
            "noise_strength": 0.2,
            "anchor_weight": 0.0,
            "residual_scale": 0.5,
            "bicubic_blend": 0.0,
            "use_orthogonal": True,
            "use_pathology_gate": True,
            "self_ensemble": False,
        },
        {
            "name": "with bicubic blend",
            "omega": 3.0,
            "noise_strength": 0.2,
            "anchor_weight": 0.4,
            "residual_scale": 0.5,
            "bicubic_blend": 0.15,
            "use_orthogonal": True,
            "use_pathology_gate": True,
            "self_ensemble": False,
        },
    ]
    if include_self_ensemble:
        variants.append(
            {
                "name": "with self-ensemble",
                "omega": 3.0,
                "noise_strength": 0.2,
                "anchor_weight": 0.4,
                "residual_scale": 0.5,
                "bicubic_blend": 0.0,
                "use_orthogonal": True,
                "use_pathology_gate": True,
                "self_ensemble": True,
            }
        )
    return variants


def load_model(cfg, checkpoint, device):
    model = SteeringSRUNet(**cfg["model"]).to(device)
    payload = torch.load(checkpoint, map_location=device)
    weight_key = "ema_model" if "ema_model" in payload else "model"
    model.load_state_dict(payload[weight_key])
    model.eval()
    return model


@torch.no_grad()
def sample_batch(diffusion, model, lr_up, scale, variant, sampling_steps, gate_config):
    def sample_once(condition):
        return diffusion.ddim_steered_sample(
            model,
            condition,
            omega=variant["omega"],
            sampling_steps=sampling_steps,
            data_consistency_weight=1.0,
            init_mode="lr_noise",
            noise_strength=variant["noise_strength"],
            gate_config=gate_config,
            degradation_scale=scale,
            anchor_weight=variant["anchor_weight"],
            residual_scale=variant["residual_scale"],
            residual_clip=0.15,
            use_orthogonal=variant["use_orthogonal"],
            use_pathology_gate=variant["use_pathology_gate"],
        )

    if variant.get("self_ensemble", False):
        preds = [
            sample_once(lr_up),
            torch.flip(sample_once(torch.flip(lr_up, dims=[-1])), dims=[-1]),
            torch.flip(sample_once(torch.flip(lr_up, dims=[-2])), dims=[-2]),
            torch.flip(sample_once(torch.flip(lr_up, dims=[-2, -1])), dims=[-2, -1]),
        ]
        sr = torch.stack(preds, dim=0).mean(dim=0)
    else:
        sr = sample_once(lr_up)
    if variant["bicubic_blend"] > 0:
        sr = ((1.0 - variant["bicubic_blend"]) * sr + variant["bicubic_blend"] * lr_up).clamp(0.0, 1.0)
    return sr.clamp(0.0, 1.0)


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
    checkpoint = cfg["inference"][f"checkpoint_x{scale}"]
    model = load_model(cfg, checkpoint, device)
    diffusion = GaussianDiffusion(**cfg["diffusion"]).to(device)
    gate_config = cfg["inference"].get("gate_config", None)

    rows = []
    safe_name = variant["name"].replace("/", "wo").replace(" ", "_")
    progress = tqdm(loader, desc=f"x{scale} {variant['name']}")
    for batch in progress:
        names = batch["name"]
        lr_up = batch["lr_up"].to(device, non_blocking=True)
        sr = sample_batch(diffusion, model, lr_up, scale, variant, args.sampling_steps, gate_config)
        for index, name in enumerate(names):
            if not bool(batch["has_hr"][index]):
                continue
            sr_np = tensor_to_numpy(sr[index])
            hr_np = tensor_to_numpy(batch["hr"][index])
            row = {
                "name": name,
                "variant": variant["name"],
                "scale": scale,
                "omega": variant["omega"],
                "noise_strength": variant["noise_strength"],
                "anchor_weight": variant["anchor_weight"],
                "residual_scale": variant["residual_scale"],
                "bicubic_blend": variant["bicubic_blend"],
                "use_orthogonal": variant["use_orthogonal"],
                "use_pathology_gate": variant["use_pathology_gate"],
                "self_ensemble": variant.get("self_ensemble", False),
            }
            row.update(compute_metrics(sr_np, hr_np))
            rows.append(row)

    out_dir = ensure_dir(Path(args.output_dir) / f"x{scale}")
    summary = save_metric_tables(
        rows,
        out_dir / f"{safe_name}_per_image.csv",
        out_dir / f"{safe_name}_summary.csv",
    )
    return {
        "variant": variant["name"],
        "scale": scale,
        **summary,
    }


def write_master_table(rows, output_dir):
    output_dir = ensure_dir(output_dir)
    fieldnames = [
        "variant",
        "x2_psnr",
        "x2_ssim",
        "x4_psnr",
        "x4_ssim",
        "x2_mae",
        "x4_mae",
        "x2_rmse",
        "x4_rmse",
    ]
    by_variant = {}
    for row in rows:
        item = by_variant.setdefault(row["variant"], {"variant": row["variant"]})
        prefix = f"x{row['scale']}"
        item[f"{prefix}_psnr"] = row.get("psnr_mean")
        item[f"{prefix}_ssim"] = row.get("ssim_mean")
        item[f"{prefix}_mae"] = row.get("mae_mean")
        item[f"{prefix}_rmse"] = row.get("rmse_mean")
    table_path = output_dir / "ablation_master_table.csv"
    with open(table_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for item in by_variant.values():
            writer.writerow(item)
    return table_path


def main():
    args = parse_args()
    seed_everything(42)
    cfg = load_config(args.config)
    device = get_device()
    output_dir = ensure_dir(args.output_dir)
    all_rows = []
    variants = build_variants(args.include_self_ensemble)
    for scale in args.scales:
        for variant in variants:
            all_rows.append(run_variant(cfg, scale, variant, args, device))
    table_path = write_master_table(all_rows, output_dir)
    print(f"Ablation done: {table_path}")


if __name__ == "__main__":
    main()
