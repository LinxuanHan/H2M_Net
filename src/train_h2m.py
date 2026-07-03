from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import HumanHRDataset
from src.h2m_net import ElasticDeformer, H2MSteeredDiffusion
from src.model import SteeringSRUNet
from src.utils import ensure_dir, load_config, save_png, seed_everything, tensor_to_numpy


def parse_args():
    parser = argparse.ArgumentParser(description="Train H2M-Net with target-free elastic manifold decoupling.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--no_preview", action="store_true")
    return parser.parse_args()


def update_ema(ema_model, model, decay):
    with torch.no_grad():
        model_state = model.state_dict()
        for name, ema_param in ema_model.state_dict().items():
            ema_param.copy_(ema_param * decay + model_state[name] * (1.0 - decay))


@torch.no_grad()
def save_preview(h2m, batch, preview_dir, epoch, sampling_steps, device):
    h2m.eval()
    lr_up = h2m.degrade(batch["hr"][:1].to(device))
    sr = h2m.sample(lr_up, sampling_steps=sampling_steps, noise_strength=0.2, omega=1.0, tau=0.1, gamma=0.02)
    save_png(tensor_to_numpy(batch["hr"][0]), Path(preview_dir) / f"epoch_{epoch:04d}_hr.png")
    save_png(tensor_to_numpy(lr_up[0]), Path(preview_dir) / f"epoch_{epoch:04d}_lr_up.png")
    save_png(tensor_to_numpy(sr[0]), Path(preview_dir) / f"epoch_{epoch:04d}_sr.png")
    h2m.train()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]
    scale = int(data_cfg["scale"])
    epochs = int(args.epochs or train_cfg["epochs"])
    batch_size = int(args.batch_size or train_cfg["batch_size"])
    seed_everything(int(train_cfg.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = HumanHRDataset(data_cfg["human_hr_dir"], scale=scale, limit=args.limit)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=int(train_cfg.get("num_workers", 0)) > 0,
    )

    model = SteeringSRUNet(**cfg["model"]).to(device)
    ema_model = SteeringSRUNet(**cfg["model"]).to(device)
    ema_model.load_state_dict(model.state_dict())
    ema_model.eval()

    h2m = H2MSteeredDiffusion(
        model,
        scale=scale,
        timesteps=int(cfg["diffusion"].get("timesteps", 1000)),
        degradation_mode=cfg["diffusion"].get("degradation_mode", "bicubic"),
    ).to(device)
    h2m_ema = H2MSteeredDiffusion(
        ema_model,
        scale=scale,
        timesteps=int(cfg["diffusion"].get("timesteps", 1000)),
        degradation_mode=cfg["diffusion"].get("degradation_mode", "bicubic"),
    ).to(device)
    deformer = ElasticDeformer(
        scale=scale,
        alpha=float(train_cfg.get("elastic_alpha", 0.08)),
        grid_size=int(train_cfg.get("elastic_grid_size", 6)),
        degradation_mode=cfg["diffusion"].get("degradation_mode", "bicubic"),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["lr"]))
    ema_decay = float(train_cfg.get("ema_decay", 0.999))

    checkpoint_dir = ensure_dir(cfg["output"]["checkpoint_dir"])
    log_dir = ensure_dir(cfg["output"]["log_dir"])
    preview_dir = ensure_dir(cfg["output"]["preview_dir"])
    log_path = log_dir / "train_log.csv"
    best_loss = float("inf")
    start_epoch = 1

    if args.resume:
        payload = torch.load(args.resume, map_location=device)
        model.load_state_dict(payload["model"])
        if "ema_model" in payload:
            ema_model.load_state_dict(payload["ema_model"])
        if "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        start_epoch = int(payload.get("epoch", 0)) + 1
        best_loss = float(payload.get("best_loss", payload.get("loss", best_loss)))

    with open(log_path, "a" if args.resume else "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", "loss", "loss_prior", "loss_follower", "loss_corrector"])
        if not args.resume:
            writer.writeheader()

    for epoch in range(start_epoch, epochs + 1):
        h2m.train()
        running = []
        running_prior = []
        running_follower = []
        running_corrector = []
        progress = tqdm(loader, desc=f"h2m epoch {epoch}/{epochs}")
        for batch in progress:
            x_hr = batch["hr"].to(device, non_blocking=True)
            loss, logs = h2m.compute_training_loss(
                x_hr,
                deformer,
                lambda_follower=float(train_cfg.get("lambda_follower", 1.0)),
                lambda_corrector=float(train_cfg.get("lambda_corrector", 1.0)),
                cond_drop_prob=float(train_cfg.get("cond_drop_prob", 0.1)),
                x0_loss_weight=float(cfg["diffusion"].get("x0_loss_weight", 0.0)),
                gradient_loss_weight=float(cfg["diffusion"].get("gradient_loss_weight", 0.0)),
                low_t_prob=float(cfg["diffusion"].get("low_t_prob", 0.0)),
                low_t_max_fraction=float(cfg["diffusion"].get("low_t_max_fraction", 0.35)),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_clip = train_cfg.get("grad_clip")
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            optimizer.step()
            update_ema(ema_model, model, ema_decay)

            running.append(float(loss.detach().cpu()))
            running_prior.append(logs["loss_prior"])
            running_follower.append(logs["loss_follower"])
            running_corrector.append(logs["loss_corrector"])
            progress.set_postfix(loss=sum(running) / len(running))

        epoch_loss = sum(running) / max(1, len(running))
        row = {
            "epoch": epoch,
            "loss": epoch_loss,
            "loss_prior": sum(running_prior) / max(1, len(running_prior)),
            "loss_follower": sum(running_follower) / max(1, len(running_follower)),
            "loss_corrector": sum(running_corrector) / max(1, len(running_corrector)),
        }
        with open(log_path, "a", newline="", encoding="utf-8") as file:
            csv.DictWriter(file, fieldnames=list(row.keys())).writerow(row)

        payload = {
            "model": model.state_dict(),
            "ema_model": ema_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg,
            "epoch": epoch,
            "loss": epoch_loss,
            "best_loss": min(best_loss, epoch_loss),
            "scale": scale,
            "method": "H2M-Net target-free elastic ORS/PSG/LRA",
        }
        torch.save(payload, checkpoint_dir / f"last_x{scale}.pt")
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(payload, checkpoint_dir / f"best_x{scale}.pt")
        if epoch == 1 or epoch % int(train_cfg.get("save_interval", 20)) == 0:
            torch.save(payload, checkpoint_dir / f"epoch_{epoch:04d}_x{scale}.pt")
            if not args.no_preview:
                h2m_ema.unet.load_state_dict(ema_model.state_dict())
                save_preview(h2m_ema, next(iter(loader)), preview_dir, epoch, int(cfg["diffusion"].get("sampling_steps", 50)), device)

    print(f"H2M-Net training done. best_loss={best_loss:.6f}, checkpoint={checkpoint_dir / f'best_x{scale}.pt'}")


if __name__ == "__main__":
    main()
