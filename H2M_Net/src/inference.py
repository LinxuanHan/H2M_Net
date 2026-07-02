import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import MouseInferenceDataset
from src.diffusion import GaussianDiffusion
from src.metrics import compute_metrics, save_metric_tables
from src.model import SteeringSRUNet
from src.utils import ensure_dir, get_device, load_config, save_nifti, save_npy, save_png, seed_everything, tensor_to_numpy
from src.vdp_vhi import compute_vdp_vhi, save_vdp_vhi
from src.visualize import save_comparison


def parse_args():
    parser = argparse.ArgumentParser(description="Zero-shot mouse SR inference with steered diffusion.")
    parser.add_argument("--config", default="configs/infer_mouse.yaml")
    parser.add_argument("--scale", type=int, default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--guidance_scale", type=float, default=None)
    parser.add_argument("--omega", type=float, default=None)
    parser.add_argument("--sampling_steps", type=int, default=None)
    parser.add_argument("--data_consistency_weight", type=float, default=None)
    parser.add_argument("--init_mode", default=None, choices=["lr_noise", "random"])
    parser.add_argument("--noise_strength", type=float, default=None)
    parser.add_argument("--anchor_weight", type=float, default=None)
    parser.add_argument("--residual_scale", type=float, default=None)
    parser.add_argument("--residual_clip", type=float, default=None)
    parser.add_argument("--bicubic_blend", type=float, default=None)
    parser.add_argument("--self_ensemble", action="store_true")
    parser.add_argument("--use_raw_weights", action="store_true")
    parser.add_argument("--disable_orthogonal", action="store_true")
    parser.add_argument("--disable_pathology_gate", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    return parser.parse_args()


def _single_item(batch, key):
    value = batch[key]
    if isinstance(value, list):
        return value[0]
    return value


def main():
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(42)
    device = get_device()
    infer_cfg = cfg["inference"]
    scale = int(args.scale or infer_cfg.get("scale", 2))
    guidance_scale = float(args.guidance_scale if args.guidance_scale is not None else infer_cfg.get("guidance_scale", 1.5))
    sampling_steps = int(args.sampling_steps or infer_cfg.get("sampling_steps", 50))
    data_consistency_weight = float(
        args.data_consistency_weight if args.data_consistency_weight is not None else infer_cfg.get("data_consistency_weight", 1.0)
    )
    init_mode = args.init_mode or infer_cfg.get("init_mode", "lr_noise")
    noise_strength = float(args.noise_strength if args.noise_strength is not None else infer_cfg.get("noise_strength", 0.35))
    anchor_weight = float(args.anchor_weight if args.anchor_weight is not None else infer_cfg.get("anchor_weight", 0.25))
    residual_scale = float(args.residual_scale if args.residual_scale is not None else infer_cfg.get("residual_scale", 0.7))
    residual_clip = float(args.residual_clip if args.residual_clip is not None else infer_cfg.get("residual_clip", 0.2))
    bicubic_blend = float(args.bicubic_blend if args.bicubic_blend is not None else infer_cfg.get("bicubic_blend", 0.15))
    self_ensemble = bool(args.self_ensemble or infer_cfg.get("self_ensemble", False))
    omega = float(args.omega if args.omega is not None else infer_cfg.get("omega", 3.0))
    checkpoint = args.checkpoint or infer_cfg[f"checkpoint_x{scale}"]

    data_cfg = cfg["data"]
    dataset = MouseInferenceDataset(
        lr_dir=data_cfg[f"mouse_lr_x{scale}_dir"],
        hr_dir=data_cfg.get(f"mouse_hr_x{scale}_dir"),
        mask_dir=data_cfg.get("mouse_mask_dir"),
        scale=scale,
        limit=args.limit,
    )
    if args.num_shards > 1:
        dataset.lr_paths = dataset.lr_paths[args.shard_id :: args.num_shards]
        if len(dataset) == 0:
            print(f"Shard {args.shard_id}/{args.num_shards} has no samples; skip.")
            return
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    model = SteeringSRUNet(**cfg["model"]).to(device)
    payload = torch.load(checkpoint, map_location=device)
    weight_key = "model" if args.use_raw_weights or "ema_model" not in payload else "ema_model"
    model.load_state_dict(payload[weight_key])
    model.eval()
    diffusion = GaussianDiffusion(**cfg["diffusion"]).to(device)

    tag = f"x{scale}_om{omega:g}_ns{noise_strength:g}_aw{anchor_weight:g}_rs{residual_scale:g}_bb{bicubic_blend:g}"
    sr_dir = ensure_dir(Path(cfg["output"]["sr_dir"]) / tag)
    metric_dir = ensure_dir(cfg["output"]["metric_dir"])
    figure_dir = ensure_dir(Path(cfg["output"]["figure_dir"]) / tag)
    shard_suffix = f"_shard{args.shard_id:02d}of{args.num_shards:02d}" if args.num_shards > 1 else ""
    metric_rows = []
    vdp_rows = []

    for batch in tqdm(loader, desc=f"infer x{scale} g={guidance_scale:g}"):
        name = _single_item(batch, "name")
        lr_up = batch["lr_up"].to(device)
        scale_tensor = batch["scale"].to(device)
        def sample_once(condition):
            # 核心创新：调用三路正交 steering 推理，而不是常规二路 CFG。
            return diffusion.ddim_steered_sample(
                model,
                condition,
                omega=omega,
                sampling_steps=sampling_steps,
                data_consistency_weight=data_consistency_weight,
                init_mode=init_mode,
                noise_strength=noise_strength,
                gate_config=infer_cfg.get("gate_config", None),
                degradation_scale=scale,
                anchor_weight=anchor_weight,
                residual_scale=residual_scale,
                residual_clip=residual_clip,
                use_orthogonal=not args.disable_orthogonal,
                use_pathology_gate=not args.disable_pathology_gate,
            )

        if self_ensemble:
            preds = [
                sample_once(lr_up),
                torch.flip(sample_once(torch.flip(lr_up, dims=[-1])), dims=[-1]),
                torch.flip(sample_once(torch.flip(lr_up, dims=[-2])), dims=[-2]),
                torch.flip(sample_once(torch.flip(lr_up, dims=[-2, -1])), dims=[-2, -1]),
            ]
            sr = torch.stack(preds, dim=0).mean(dim=0)
        else:
            sr = sample_once(lr_up)
        if bicubic_blend > 0:
            sr = ((1.0 - bicubic_blend) * sr + bicubic_blend * lr_up).clamp(0.0, 1.0)
        sr_np = tensor_to_numpy(sr[0])
        lr_up_np = tensor_to_numpy(lr_up[0])
        if infer_cfg.get("save_png", True):
            save_png(sr_np, sr_dir / f"{name}_sr.png")
        if infer_cfg.get("save_npy", True):
            save_npy(sr_np, sr_dir / f"{name}_sr.npy")
        if infer_cfg.get("save_nifti", False):
            save_nifti(sr_np, sr_dir / f"{name}_sr.nii.gz")

        has_hr = bool(batch["has_hr"][0])
        hr_np = tensor_to_numpy(batch["hr"][0]) if has_hr else None
        mask_np = tensor_to_numpy(batch["mask"][0]) if bool(batch["has_mask"][0]) else None
        if has_hr and cfg["evaluation"].get("compute_ssim_psnr", True):
            row = {
                "name": name,
                "scale": scale,
                "guidance_scale": guidance_scale,
                "omega": omega,
                "noise_strength": noise_strength,
                "anchor_weight": anchor_weight,
                "residual_scale": residual_scale,
                "bicubic_blend": bicubic_blend,
                "self_ensemble": self_ensemble,
            }
            row.update(compute_metrics(sr_np, hr_np, mask_np if cfg["evaluation"].get("use_mask", True) else None))
            metric_rows.append(row)
        if cfg["evaluation"].get("compute_vdp_vhi", True):
            vdp_cfg = cfg.get("vdp_vhi", {})
            row = {"name": name, "scale": scale, "omega": omega, "noise_strength": noise_strength}
            row.update(
                compute_vdp_vhi(
                    sr_np,
                    mask_np,
                    mode=vdp_cfg.get("defect_threshold_mode", "relative_to_mean"),
                    relative_threshold=float(vdp_cfg.get("defect_relative_threshold", 0.6)),
                    percentile=float(vdp_cfg.get("defect_percentile", 15)),
                    fixed_threshold=float(vdp_cfg.get("fixed_threshold", 0.1)),
                )
            )
            vdp_rows.append(row)
        if len(metric_rows) <= 8:
            save_comparison(lr_up_np, sr_np, hr_np, figure_dir / f"{name}_comparison.png", title=f"{name} x{scale}")

    if metric_rows:
        summary = save_metric_tables(
            metric_rows,
            metric_dir / f"steered_diffusion_{tag}{shard_suffix}_per_image.csv",
            metric_dir / f"steered_diffusion_{tag}{shard_suffix}_summary.csv",
        )
        print(summary)
    if vdp_rows:
        save_vdp_vhi(vdp_rows, metric_dir / f"steered_diffusion_{tag}{shard_suffix}_vdp_vhi.csv")
    print(f"Inference done. results={sr_dir}")


if __name__ == "__main__":
    main()
