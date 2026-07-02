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
        "PyTorch is required for DDRM. Run this script in the conda environment "
        "that has torch installed."
    ) from error


DATASETS = {
    "human_train": ("human", "train"),
    "mouse_test": ("mouse", "test"),
}


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


class UnconditionalDenoiser(nn.Module):
    def __init__(self, channels=64, blocks=8):
        super().__init__()
        self.in_conv = nn.Conv2d(2, channels, 3, padding=1)
        self.blocks = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])
        self.out = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, 1, 3, padding=1),
        )

    def forward(self, x, t_normalized):
        time_map = torch.ones_like(x) * t_normalized.view(-1, 1, 1, 1)
        x = torch.cat([x, time_map], dim=1)
        x = self.in_conv(x)
        x = self.blocks(x)
        return self.out(x)


def parse_args():
    parser = argparse.ArgumentParser(description="Run DDRM-style zero-shot diffusion SR.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-root", type=Path, default=Path("experiments") / "ddrm_sr")
    parser.add_argument("--scales", type=int, nargs="+", default=[2, 4], choices=[2, 4])
    parser.add_argument("--train-steps", type=int, default=12000)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--sample-steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--consistency-weight", type=float, default=1.0)
    parser.add_argument("--init-noise-std", type=float, default=0.0)
    parser.add_argument("--prior-strength", type=float, default=0.2)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


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


def image_to_tensor(image, device):
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0).unsqueeze(0).to(device)


def tensor_to_image(tensor):
    array = tensor.detach().squeeze().clamp(0.0, 1.0).cpu().numpy()
    array = np.rint(array * 255.0).astype(np.uint8)
    return Image.fromarray(array)


def collect_raw_images(data_root, set_name):
    species, split = DATASETS[set_name]
    raw_dir = data_root / species / split / "raw"
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Missing raw directory: {raw_dir}")
    paths = sorted(raw_dir.glob("*.png"))
    if not paths:
        raise RuntimeError(f"No PNG images found in {raw_dir}")
    return paths


def collect_pairs(data_root, scale):
    raw_dir = data_root / "mouse" / "test" / "raw"
    lr_dir = data_root / "mouse" / "test" / f"down_{scale}x_bilinear"
    pairs = []
    for hr_path in sorted(raw_dir.glob("*.png")):
        lr_path = lr_dir / hr_path.name
        if not lr_path.is_file():
            raise FileNotFoundError(f"Missing LR pair for {hr_path}: {lr_path}")
        pairs.append((lr_path, hr_path))
    if not pairs:
        raise RuntimeError("No mouse test pairs found")
    return pairs


def make_beta_schedule(steps, device):
    betas = torch.linspace(1e-4, 2e-2, steps, device=device)
    alphas = 1.0 - betas
    return torch.cumprod(alphas, dim=0)


def load_training_images(data_root, device):
    tensors = []
    for path in collect_raw_images(data_root, "human_train"):
        tensors.append(image_to_tensor(load_grayscale(path), device))
    return tensors


def sample_training_batch(images, batch_size):
    batch = []
    for _ in range(batch_size):
        image = random.choice(images)
        if random.random() < 0.5:
            image = torch.flip(image, dims=[-1])
        if random.random() < 0.5:
            image = torch.flip(image, dims=[-2])
        batch.append(image)
    return torch.cat(batch, dim=0)


def train_prior(args, device):
    images = load_training_images(args.data_root, device)
    model = UnconditionalDenoiser(args.channels, args.blocks).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    alpha_bars = make_beta_schedule(args.diffusion_steps, device)

    model.train()
    for step in range(1, args.train_steps + 1):
        clean = sample_training_batch(images, args.batch_size)
        batch_size = clean.shape[0]
        t = torch.randint(0, args.diffusion_steps, (batch_size,), device=device)
        noise = torch.randn_like(clean)
        alpha_bar = alpha_bars[t].view(-1, 1, 1, 1)
        noisy = alpha_bar.sqrt() * clean + (1.0 - alpha_bar).sqrt() * noise
        t_normalized = t.float() / max(1, args.diffusion_steps - 1)
        prediction = model(noisy, t_normalized)
        loss = F.mse_loss(prediction, noise)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step % 1000 == 0 or step == args.train_steps:
            print(f"prior train step {step}/{args.train_steps}: loss={float(loss.detach().cpu()):.6f}")
    return model


def make_schedule(diffusion_steps, sample_steps):
    values = np.linspace(diffusion_steps - 1, 0, sample_steps)
    values = np.unique(np.rint(values).astype(np.int64))[::-1]
    if values[-1] != 0:
        values = np.concatenate([values, np.array([0], dtype=np.int64)])
    return [int(v) for v in values]


def resize_tensor(tensor, size):
    return F.interpolate(tensor, size=size, mode="bicubic", align_corners=False).clamp(0.0, 1.0)


def project_data_consistency(x0, lr_tensor, consistency_weight):
    lr_prediction = resize_tensor(x0, lr_tensor.shape[-2:])
    correction = resize_tensor(lr_tensor - lr_prediction, x0.shape[-2:])
    return (x0 + consistency_weight * correction).clamp(0.0, 1.0)


def ddrm_sample(model, lr_tensor, hr_size, alpha_bars, timesteps, args):
    condition = resize_tensor(lr_tensor, (hr_size[1], hr_size[0]))
    x = condition + torch.randn_like(condition) * args.init_noise_std
    x = x.clamp(0.0, 1.0)
    model.eval()
    with torch.no_grad():
        for index, t_value in enumerate(timesteps):
            t = torch.full((1,), t_value, device=lr_tensor.device, dtype=torch.long)
            t_normalized = t.float() / max(1, len(alpha_bars) - 1)
            alpha_t = alpha_bars[t].view(1, 1, 1, 1)
            eps = model(x, t_normalized)
            x0 = (x - (1.0 - alpha_t).sqrt() * eps) / alpha_t.sqrt().clamp_min(1e-8)
            x0 = project_data_consistency(x0, lr_tensor, args.consistency_weight)
            x0 = (condition + args.prior_strength * (x0 - condition)).clamp(0.0, 1.0)
            if index == len(timesteps) - 1:
                x = x0
            else:
                next_t = timesteps[index + 1]
                alpha_next = alpha_bars[next_t].view(1, 1, 1, 1)
                x = alpha_next.sqrt() * x0 + (1.0 - alpha_next).sqrt() * eps
                x = x.clamp(0.0, 1.0)
    return x.clamp(0.0, 1.0)


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


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["set", "scale", "filename", "width", "height", "sample_steps", "psnr", "ssim", "mse", "mae"]
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


def evaluate_scale(args, model, scale, device):
    pairs = collect_pairs(args.data_root, scale)
    if args.limit > 0:
        pairs = pairs[: args.limit]
    alpha_bars = make_beta_schedule(args.diffusion_steps, device)
    timesteps = make_schedule(args.diffusion_steps, args.sample_steps)

    result_root = args.output_root / f"x{scale}" / "mouse_test"
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
            lr_tensor = image_to_tensor(load_grayscale(lr_path), device)
            sr_tensor = ddrm_sample(model, lr_tensor, hr_image.size, alpha_bars, timesteps, args)
            sr_image = tensor_to_image(sr_tensor)
            sr_image.save(sr_path)
        metrics = compute_metrics(sr_image, hr_image)
        rows.append(
            {
                "set": "mouse_test",
                "scale": scale,
                "filename": hr_path.name,
                "width": hr_image.width,
                "height": hr_image.height,
                "sample_steps": len(timesteps),
                **metrics,
            }
        )
        if index == 1 or index % 100 == 0 or index == len(pairs):
            print(
                f"x{scale} mouse_test [{index}/{len(pairs)}] {hr_path.name}: "
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
    config["checkpoint"] = str(args.checkpoint) if args.checkpoint is not None else None
    config["device"] = str(device)
    with (args.output_root / "config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)

    checkpoint_dir = args.output_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model = UnconditionalDenoiser(args.channels, args.blocks).to(device)
    checkpoint_path = args.checkpoint
    if args.skip_train:
        if checkpoint_path is None:
            checkpoint_path = args.output_root / "checkpoints" / "unconditional_hr_prior.pt"
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        model = train_prior(args, device)
        torch.save(model.state_dict(), checkpoint_dir / "unconditional_hr_prior.pt")

    all_summaries = {}
    for scale in args.scales:
        summary = evaluate_scale(args, model, scale, device)
        all_summaries[f"x{scale}/mouse_test"] = summary
        print(
            f"x{scale} mouse_test: "
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
