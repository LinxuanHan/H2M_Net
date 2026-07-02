import argparse
import csv
import json
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
        "PyTorch is required for DIP. Run this script in the conda environment "
        "that has torch installed."
    ) from error


DATASETS = {
    "human_val": ("human", "val"),
    "mouse_test": ("mouse", "test"),
}


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DIPUNet(nn.Module):
    def __init__(self, in_channels=32, base_channels=32):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 4)
        self.dec3 = ConvBlock(base_channels * 8, base_channels * 2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels)
        self.dec1 = ConvBlock(base_channels * 2, base_channels)
        self.out = nn.Conv2d(base_channels, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        e3 = self.enc3(F.avg_pool2d(e2, 2))
        b = self.bottleneck(F.avg_pool2d(e3, 2))

        d3 = F.interpolate(b, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return torch.sigmoid(self.out(d1))


def parse_args():
    parser = argparse.ArgumentParser(description="Run Medical DIP super-resolution baseline.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-root", type=Path, default=Path("experiments") / "dip")
    parser.add_argument("--scales", type=int, nargs="+", default=[2, 4], choices=[2, 4])
    parser.add_argument(
        "--sets",
        nargs="+",
        default=["mouse_test"],
        choices=sorted(DATASETS.keys()),
    )
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--noise-channels", type=int, default=32)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--tv-weight", type=float, default=1e-5)
    parser.add_argument("--anchor-weight", type=float, default=5e-2)
    parser.add_argument("--input-noise-std", type=float, default=0.03)
    parser.add_argument("--input-mode", default="bicubic_noise", choices=["noise", "bicubic_noise"])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--save-checkpoints", action="store_true")
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


def downsample_to_lr(sr_tensor, lr_shape):
    return F.interpolate(sr_tensor, size=lr_shape, mode="bicubic", align_corners=False).clamp(0.0, 1.0)


def total_variation(image):
    vertical = torch.mean(torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]))
    horizontal = torch.mean(torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]))
    return vertical + horizontal


def optimize_single_image(lr_image, hr_size, args, device):
    lr_tensor = image_to_tensor(lr_image, device)
    height, width = hr_size[1], hr_size[0]
    model = DIPUNet(args.noise_channels, args.base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    bicubic = F.interpolate(lr_tensor, size=(height, width), mode="bicubic", align_corners=False).clamp(0.0, 1.0)
    noise = torch.rand(1, args.noise_channels, height, width, device=device)
    if args.input_mode == "bicubic_noise":
        noise[:, :1] = bicubic
    fixed_noise = noise.detach().clone()
    best_loss = float("inf")
    best_output = None

    model.train()
    for _ in range(args.steps):
        noisy_input = fixed_noise
        if args.input_noise_std > 0:
            noisy_input = fixed_noise + torch.randn_like(fixed_noise) * args.input_noise_std

        sr_tensor = model(noisy_input)
        lr_prediction = downsample_to_lr(sr_tensor, lr_tensor.shape[-2:])
        loss = F.mse_loss(lr_prediction, lr_tensor)
        if args.tv_weight > 0:
            loss = loss + args.tv_weight * total_variation(sr_tensor)
        if args.anchor_weight > 0:
            loss = loss + args.anchor_weight * F.mse_loss(sr_tensor, bicubic)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_value = float(loss.detach().cpu())
        if loss_value < best_loss:
            best_loss = loss_value
            best_output = sr_tensor.detach().clone()

    return model, best_output


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
        "steps",
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


def run_set(args, set_name, scale, device):
    pairs = collect_pairs(args.data_root, set_name, scale)
    if args.limit > 0:
        pairs = pairs[: args.limit]

    result_root = args.output_root / f"x{scale}" / set_name
    sr_dir = result_root / "sr"
    metrics_dir = result_root / "metrics"
    ckpt_dir = result_root / "checkpoints"
    sr_dir.mkdir(parents=True, exist_ok=True)
    if args.save_checkpoints:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for index, (lr_path, hr_path) in enumerate(pairs, start=1):
        hr_image = load_grayscale(hr_path)
        sr_path = sr_dir / hr_path.name

        if sr_path.exists() and not args.overwrite:
            sr_image = load_grayscale(sr_path)
        else:
            lr_image = load_grayscale(lr_path)
            model, sr_tensor = optimize_single_image(lr_image, hr_image.size, args, device)
            sr_image = tensor_to_image(sr_tensor)
            sr_image.save(sr_path)
            if args.save_checkpoints:
                torch.save(model.state_dict(), ckpt_dir / f"{hr_path.stem}.pt")

        metrics = compute_metrics(sr_image, hr_image)
        rows.append(
            {
                "set": set_name,
                "scale": scale,
                "filename": hr_path.name,
                "width": hr_image.width,
                "height": hr_image.height,
                "steps": args.steps,
                **metrics,
            }
        )
        print(
            f"x{scale} {set_name} [{index}/{len(pairs)}] {hr_path.name}: "
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
        for set_name in args.sets:
            summary = run_set(args, set_name, scale, device)
            all_summaries[f"x{scale}/{set_name}"] = summary
            print(
                f"x{scale} {set_name}: "
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
