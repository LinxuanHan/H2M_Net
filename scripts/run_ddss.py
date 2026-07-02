import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as error:
    raise SystemExit(
        "PyTorch is required for DDSS. Run this script in the conda environment "
        "that has torch installed."
    ) from error


DATASETS = {
    "human_train": ("human", "train"),
    "human_val": ("human", "val"),
    "mouse_test": ("mouse", "test"),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a DDSS-style fast sampler search for residual diffusion SR."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-root", type=Path, default=Path("experiments") / "ddss_sr")
    parser.add_argument("--scales", type=int, nargs="+", default=[2, 4], choices=[2, 4])
    parser.add_argument("--train-steps", type=int, default=12000)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--residual-range", type=float, default=0.25)
    parser.add_argument("--start-mode", default="zero", choices=["zero", "noise"])
    parser.add_argument("--search-max-images", type=int, default=32)
    parser.add_argument("--eval-set", default="mouse_test", choices=sorted(DATASETS.keys()))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x) * 0.1


class ResidualDenoiser(nn.Module):
    def __init__(self, channels=64, blocks=8):
        super().__init__()
        self.in_conv = nn.Conv2d(3, channels, 3, padding=1)
        self.blocks = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])
        self.out = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, 1, 3, padding=1),
        )

    def forward(self, noisy_residual, condition, t_normalized):
        time_map = torch.ones_like(noisy_residual) * t_normalized.view(-1, 1, 1, 1)
        x = torch.cat([noisy_residual, condition, time_map], dim=1)
        x = self.in_conv(x)
        x = self.blocks(x)
        return self.out(x)


def choose_device(name):
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_grayscale(path):
    return Image.open(path).convert("L")


def image_to_array(image):
    return np.asarray(image, dtype=np.float32) / 255.0


def array_to_image(array):
    array = np.rint(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)
    return Image.fromarray(array)


def pil_bicubic_to_hr(lr_image, hr_size):
    return lr_image.resize(hr_size, Image.Resampling.BICUBIC)


def image_to_tensor(image, device):
    return torch.from_numpy(image_to_array(image)).unsqueeze(0).unsqueeze(0).to(device)


def tensor_to_image(tensor):
    array = tensor.detach().squeeze().clamp(0.0, 1.0).cpu().numpy()
    return array_to_image(array)


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


def load_training_tensors(data_root, scale, residual_range, device):
    examples = []
    for lr_path, hr_path in collect_pairs(data_root, "human_train", scale):
        hr_image = load_grayscale(hr_path)
        lr_image = load_grayscale(lr_path)
        condition_image = pil_bicubic_to_hr(lr_image, hr_image.size)
        hr = image_to_tensor(hr_image, device)
        condition = image_to_tensor(condition_image, device)
        residual = ((hr - condition) / residual_range).clamp(-1.0, 1.0)
        examples.append((condition, residual))
    return examples


def make_beta_schedule(steps, device):
    betas = torch.linspace(1e-4, 2e-2, steps, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return alpha_bars


def sample_training_batch(examples, batch_size):
    conditions = []
    residuals = []
    for _ in range(batch_size):
        condition, residual = random.choice(examples)
        if random.random() < 0.5:
            condition = torch.flip(condition, dims=[-1])
            residual = torch.flip(residual, dims=[-1])
        if random.random() < 0.5:
            condition = torch.flip(condition, dims=[-2])
            residual = torch.flip(residual, dims=[-2])
        conditions.append(condition)
        residuals.append(residual)
    return torch.cat(conditions, dim=0), torch.cat(residuals, dim=0)


def train_model(args, scale, device):
    examples = load_training_tensors(args.data_root, scale, args.residual_range, device)
    model = ResidualDenoiser(args.channels, args.blocks).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    alpha_bars = make_beta_schedule(args.diffusion_steps, device)

    model.train()
    for step in range(1, args.train_steps + 1):
        condition, residual = sample_training_batch(examples, args.batch_size)
        batch_size = condition.shape[0]
        t = torch.randint(0, args.diffusion_steps, (batch_size,), device=device)
        noise = torch.randn_like(residual)
        alpha_bar = alpha_bars[t].view(-1, 1, 1, 1)
        noisy_residual = alpha_bar.sqrt() * residual + (1.0 - alpha_bar).sqrt() * noise
        t_normalized = t.float() / max(1, args.diffusion_steps - 1)
        prediction = model(noisy_residual, condition, t_normalized)
        loss = F.mse_loss(prediction, noise)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step % 1000 == 0 or step == args.train_steps:
            print(f"x{scale} train step {step}/{args.train_steps}: loss={float(loss.detach().cpu()):.6f}")
    return model


def make_schedule(diffusion_steps, sample_steps, kind):
    if sample_steps >= diffusion_steps:
        values = np.arange(diffusion_steps - 1, -1, -1)
    else:
        if kind == "uniform":
            values = np.linspace(diffusion_steps - 1, 0, sample_steps)
        elif kind == "quadratic":
            values = (np.linspace(math.sqrt(diffusion_steps - 1), 0, sample_steps) ** 2)
        elif kind == "late":
            values = (np.linspace((diffusion_steps - 1) ** 0.35, 0, sample_steps) ** (1 / 0.35))
        else:
            raise ValueError(f"Unknown schedule kind: {kind}")
        values = np.unique(np.rint(values).astype(np.int64))[::-1]
        if values[-1] != 0:
            values = np.concatenate([values, np.array([0], dtype=np.int64)])
    return [int(v) for v in values]


def ddim_sample_residual(model, condition, alpha_bars, timesteps, start_mode):
    if start_mode == "noise":
        x = torch.randn_like(condition)
    else:
        x = torch.zeros_like(condition)
    model.eval()
    with torch.no_grad():
        for index, t_value in enumerate(timesteps):
            t = torch.full((condition.shape[0],), t_value, device=condition.device, dtype=torch.long)
            t_normalized = t.float() / max(1, len(alpha_bars) - 1)
            alpha_t = alpha_bars[t].view(-1, 1, 1, 1)
            eps = model(x, condition, t_normalized)
            x0 = (x - (1.0 - alpha_t).sqrt() * eps) / alpha_t.sqrt().clamp_min(1e-8)
            x0 = x0.clamp(-1.0, 1.0)
            if index == len(timesteps) - 1:
                x = x0
            else:
                next_t = timesteps[index + 1]
                alpha_next = alpha_bars[next_t].view(1, 1, 1, 1)
                x = alpha_next.sqrt() * x0 + (1.0 - alpha_next).sqrt() * eps
    return x.clamp(-1.0, 1.0)


def infer_image(model, lr_image, hr_size, alpha_bars, timesteps, args, device):
    condition_image = pil_bicubic_to_hr(lr_image, hr_size)
    condition = image_to_tensor(condition_image, device)
    residual = ddim_sample_residual(model, condition, alpha_bars, timesteps, args.start_mode)
    sr = (condition + residual * args.residual_range).clamp(0.0, 1.0)
    return tensor_to_image(sr)


def compute_metrics(sr_image, hr_image):
    sr = np.asarray(sr_image, dtype=np.float32)
    hr = np.asarray(hr_image, dtype=np.float32)
    diff = sr - hr
    return {
        "psnr": float(peak_signal_noise_ratio(hr, sr, data_range=255)),
        "ssim": float(structural_similarity(hr, sr, data_range=255)),
        "mse": float(np.mean(diff * diff)),
        "mae": float(np.mean(np.abs(diff))),
    }


def evaluate_schedule(args, model, scale, alpha_bars, timesteps, device, max_images):
    pairs = collect_pairs(args.data_root, "human_val", scale)[:max_images]
    scores = []
    for lr_path, hr_path in pairs:
        hr_image = load_grayscale(hr_path)
        lr_image = load_grayscale(lr_path)
        sr_image = infer_image(model, lr_image, hr_image.size, alpha_bars, timesteps, args, device)
        scores.append(compute_metrics(sr_image, hr_image)["psnr"])
    return float(np.mean(scores))


def search_sampler(args, model, scale, device):
    alpha_bars = make_beta_schedule(args.diffusion_steps, device)
    candidates = []
    for sample_steps in [5, 10, 20, 30, 50]:
        for kind in ["uniform", "quadratic", "late"]:
            timesteps = make_schedule(args.diffusion_steps, sample_steps, kind)
            psnr = evaluate_schedule(
                args,
                model,
                scale,
                alpha_bars,
                timesteps,
                device,
                args.search_max_images,
            )
            candidates.append(
                {
                    "kind": kind,
                    "requested_steps": sample_steps,
                    "actual_steps": len(timesteps),
                    "timesteps": timesteps,
                    "human_val_psnr": psnr,
                }
            )
            print(f"x{scale} search {kind}/{sample_steps}: PSNR={psnr:.4f}, steps={len(timesteps)}")
    best = max(candidates, key=lambda item: item["human_val_psnr"])
    return best, candidates


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "set",
        "scale",
        "filename",
        "width",
        "height",
        "sampler",
        "sample_steps",
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
    return {
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


def evaluate_test(args, model, scale, best_sampler, device):
    pairs = collect_pairs(args.data_root, args.eval_set, scale)
    if args.limit > 0:
        pairs = pairs[: args.limit]
    alpha_bars = make_beta_schedule(args.diffusion_steps, device)
    timesteps = best_sampler["timesteps"]

    result_root = args.output_root / f"x{scale}" / args.eval_set
    sr_dir = result_root / "sr"
    metrics_dir = result_root / "metrics"
    sr_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for index, (lr_path, hr_path) in enumerate(pairs, start=1):
        hr_image = load_grayscale(hr_path)
        sr_path = sr_dir / hr_path.name
        if sr_path.exists() and not args.overwrite:
            sr_image = load_grayscale(sr_path)
        else:
            lr_image = load_grayscale(lr_path)
            sr_image = infer_image(model, lr_image, hr_image.size, alpha_bars, timesteps, args, device)
            sr_image.save(sr_path)
        metrics = compute_metrics(sr_image, hr_image)
        rows.append(
            {
                "set": args.eval_set,
                "scale": scale,
                "filename": hr_path.name,
                "width": hr_image.width,
                "height": hr_image.height,
                "sampler": best_sampler["kind"],
                "sample_steps": best_sampler["actual_steps"],
                **metrics,
            }
        )
        if index == 1 or index % 100 == 0 or index == len(pairs):
            print(
                f"x{scale} {args.eval_set} [{index}/{len(pairs)}] {hr_path.name}: "
                f"PSNR={metrics['psnr']:.4f}, SSIM={metrics['ssim']:.4f}"
            )

    summary = summarize(rows)
    write_csv(metrics_dir / "per_image_metrics.csv", rows)
    with (metrics_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    return summary


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = choose_device(args.device)
    args.output_root.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config["data_root"] = str(args.data_root)
    config["output_root"] = str(args.output_root)
    config["device"] = str(device)
    with (args.output_root / "config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)

    all_summaries = {}
    for scale in args.scales:
        model = train_model(args, scale, device)
        checkpoint_dir = args.output_root / f"x{scale}" / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_dir / "residual_diffusion.pt")

        best_sampler, candidates = search_sampler(args, model, scale, device)
        search_dir = args.output_root / f"x{scale}" / "sampler_search"
        search_dir.mkdir(parents=True, exist_ok=True)
        with (search_dir / "candidates.json").open("w", encoding="utf-8") as file:
            json.dump(candidates, file, indent=2)
        with (search_dir / "best_sampler.json").open("w", encoding="utf-8") as file:
            json.dump(best_sampler, file, indent=2)

        summary = evaluate_test(args, model, scale, best_sampler, device)
        all_summaries[f"x{scale}/{args.eval_set}"] = summary
        print(
            f"x{scale} {args.eval_set}: "
            f"n={summary['count']}, "
            f"PSNR={summary['psnr_mean']:.4f}, "
            f"SSIM={summary['ssim_mean']:.4f}, "
            f"MSE={summary['mse_mean']:.4f}, "
            f"MAE={summary['mae_mean']:.4f}"
        )

    with (args.output_root / "summary_all.json").open("w", encoding="utf-8") as file:
        json.dump(all_summaries, file, indent=2)


if __name__ == "__main__":
    main()
