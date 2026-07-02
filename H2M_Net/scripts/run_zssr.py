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
        "PyTorch is required for ZSSR. Install torch or run this script in a "
        "Python environment that already has torch available."
    ) from error


DATASETS = {
    "human_val": ("human", "val"),
    "mouse_test": ("mouse", "test"),
}


class ZSSRNet(nn.Module):
    def __init__(self, channels=64, depth=8):
        super().__init__()
        layers = [nn.Conv2d(1, channels, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers.extend([nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(inplace=True)])
        layers.append(nn.Conv2d(channels, 1, 3, padding=1))
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return torch.clamp(x + self.body(x), 0.0, 1.0)


def parse_args():
    parser = argparse.ArgumentParser(description="Run image-specific ZSSR baseline.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-root", type=Path, default=Path("experiments") / "zssr")
    parser.add_argument("--scales", type=int, nargs="+", default=[2, 4], choices=[2, 4])
    parser.add_argument(
        "--sets",
        nargs="+",
        default=["human_val", "mouse_test"],
        choices=sorted(DATASETS.keys()),
    )
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--crop-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--depth", type=int, default=8)
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
    return Image.fromarray(array, mode="L")


def resize_pil(image, size, resample=Image.Resampling.BICUBIC):
    return image.resize(size, resample)


def tensor_resize(tensor, size):
    return F.interpolate(tensor, size=size, mode="bicubic", align_corners=False).clamp(0.0, 1.0)


def random_training_pair(lr_tensor, scale, crop_size):
    _, _, height, width = lr_tensor.shape
    target_size = min(crop_size, height, width)
    target_size = max(scale, (target_size // scale) * scale)
    if target_size < scale:
        target_size = min(height, width)

    top = random.randint(0, height - target_size)
    left = random.randint(0, width - target_size)
    target = lr_tensor[:, :, top : top + target_size, left : left + target_size]
    low_size = (max(1, target_size // scale), max(1, target_size // scale))
    low = tensor_resize(target, low_size)
    input_tensor = tensor_resize(low, (target_size, target_size))

    if random.random() < 0.5:
        input_tensor = torch.flip(input_tensor, dims=[-1])
        target = torch.flip(target, dims=[-1])
    if random.random() < 0.5:
        input_tensor = torch.flip(input_tensor, dims=[-2])
        target = torch.flip(target, dims=[-2])

    rotations = random.randint(0, 3)
    if rotations:
        input_tensor = torch.rot90(input_tensor, rotations, dims=[-2, -1])
        target = torch.rot90(target, rotations, dims=[-2, -1])

    return input_tensor, target


def train_single_image(lr_image, scale, args, device):
    model = ZSSRNet(channels=args.channels, depth=args.depth).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_tensor = image_to_tensor(lr_image, device)

    model.train()
    for step in range(args.steps):
        input_tensor, target = random_training_pair(lr_tensor, scale, args.crop_size)
        prediction = model(input_tensor)
        loss = F.l1_loss(prediction, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    return model


def infer(model, lr_image, target_size, device):
    model.eval()
    with torch.no_grad():
        lr_tensor = image_to_tensor(lr_image, device)
        upsampled = tensor_resize(lr_tensor, (target_size[1], target_size[0]))
        sr_tensor = model(upsampled)
    return tensor_to_image(sr_tensor)


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
        sr_path = sr_dir / hr_path.name
        if sr_path.exists() and not args.overwrite:
            sr_image = load_grayscale(sr_path)
            hr_image = load_grayscale(hr_path)
        else:
            lr_image = load_grayscale(lr_path)
            hr_image = load_grayscale(hr_path)
            model = train_single_image(lr_image, scale, args, device)
            sr_image = infer(model, lr_image, hr_image.size, device)
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
