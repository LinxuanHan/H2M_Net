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
        "PyTorch is required for SMORE. Run this script in the conda environment "
        "that has torch installed."
    ) from error


DATASETS = {
    "human_val": ("human", "val"),
    "mouse_test": ("mouse", "test"),
}


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x) * 0.1


class SMORENet(nn.Module):
    def __init__(self, channels=64, blocks=8, residual_scale=1.0):
        super().__init__()
        self.residual_scale = residual_scale
        layers = [
            nn.Conv2d(1, channels, 3, padding=1),
            nn.ReLU(inplace=True),
        ]
        layers.extend(ResidualBlock(channels) for _ in range(blocks))
        layers.append(nn.Conv2d(channels, 1, 3, padding=1))
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return torch.clamp(x + self.body(x) * self.residual_scale, 0.0, 1.0)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a 2D self-supervised SMORE-style MRI super-resolution baseline."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-root", type=Path, default=Path("experiments") / "smore")
    parser.add_argument("--scales", type=int, nargs="+", default=[2, 4], choices=[2, 4])
    parser.add_argument(
        "--sets",
        nargs="+",
        default=["mouse_test"],
        choices=sorted(DATASETS.keys()),
    )
    parser.add_argument("--train-set", default="mouse_test", choices=sorted(DATASETS.keys()))
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--self-supervision-scale",
        type=int,
        default=0,
        choices=[0, 2, 4],
        help="Internal synthetic degradation scale. 0 uses auto mode: x2 for all requested scales.",
    )
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


def tensor_resize(tensor, size):
    return F.interpolate(tensor, size=size, mode="bicubic", align_corners=False).clamp(0.0, 1.0)


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


def load_lr_tensors(data_root, set_name, scale, device):
    pairs = collect_pairs(data_root, set_name, scale)
    return [image_to_tensor(load_grayscale(lr_path), device) for lr_path, _ in pairs]


def augment_batch(batch):
    if random.random() < 0.5:
        batch = torch.flip(batch, dims=[-1])
    if random.random() < 0.5:
        batch = torch.flip(batch, dims=[-2])
    rotations = random.randint(0, 3)
    if rotations:
        batch = torch.rot90(batch, rotations, dims=[-2, -1])
    return batch


def make_self_supervised_batch(lr_tensors, scale, batch_size):
    targets = []
    for _ in range(batch_size):
        target = random.choice(lr_tensors)
        targets.append(target)
    target_batch = torch.cat(targets, dim=0)
    target_batch = augment_batch(target_batch)
    height, width = target_batch.shape[-2:]
    low_size = (max(1, height // scale), max(1, width // scale))
    low = tensor_resize(target_batch, low_size)
    inputs = tensor_resize(low, (height, width))
    return inputs, target_batch


def train_smore(args, scale, device):
    lr_tensors = load_lr_tensors(args.data_root, args.train_set, scale, device)
    model = SMORENet(args.channels, args.blocks, args.residual_scale).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    self_scale = 2 if args.self_supervision_scale == 0 else args.self_supervision_scale

    model.train()
    for step in range(1, args.steps + 1):
        inputs, targets = make_self_supervised_batch(lr_tensors, self_scale, args.batch_size)
        prediction = model(inputs)
        loss = F.l1_loss(prediction, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 500 == 0 or step == args.steps:
            print(f"x{scale} train step {step}/{args.steps}: loss={float(loss.detach().cpu()):.6f}")
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


def evaluate_set(args, model, set_name, scale, device):
    pairs = collect_pairs(args.data_root, set_name, scale)
    if args.limit > 0:
        pairs = pairs[: args.limit]

    result_root = args.output_root / f"x{scale}" / set_name
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
            sr_image = infer(model, lr_image, hr_image.size, device)
            sr_image.save(sr_path)

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
        if index == 1 or index % 100 == 0 or index == len(pairs):
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
        model = train_smore(args, scale, device)
        checkpoint_dir = args.output_root / f"x{scale}" / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_dir / "smore.pt")

        for set_name in args.sets:
            summary = evaluate_set(args, model, set_name, scale, device)
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
