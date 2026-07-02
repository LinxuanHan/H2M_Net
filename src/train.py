import argparse
import csv
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from src.dataset import HumanHRDataset
from src.diffusion import GaussianDiffusion
from src.model import SteeringSRUNet
from src.utils import ensure_dir, get_device, load_config, save_png, seed_everything, tensor_to_numpy


def parse_args():
    parser = argparse.ArgumentParser(description="Train human-prior steered diffusion SR model.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sampling_steps", type=int, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--no_preview", action="store_true")
    return parser.parse_args()


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return True, local_rank, rank, world_size, torch.device(f"cuda:{local_rank}")


def cleanup_distributed(is_distributed):
    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()


def is_main_process(rank):
    return rank == 0


def reduce_mean(value, device, is_distributed, world_size):
    tensor = torch.tensor([value], device=device, dtype=torch.float32)
    if is_distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= world_size
    return float(tensor.item())


def update_ema(ema_model, model, decay):
    with torch.no_grad():
        model_state = model.state_dict()
        for name, ema_param in ema_model.state_dict().items():
            ema_param.copy_(ema_param * decay + model_state[name] * (1.0 - decay))


@torch.no_grad()
def save_preview(model, diffusion, batch, epoch, preview_dir, sampling_steps):
    model.eval()
    lr_up = batch["lr_up"][:1].to(next(model.parameters()).device)
    condition = torch.full((lr_up.shape[0],), 2, device=lr_up.device, dtype=torch.long)
    sr = diffusion.ddim_sample(model, lr_up, condition, sampling_steps=sampling_steps, guidance_scale=1.2, data_consistency_weight=0.3)
    save_png(tensor_to_numpy(lr_up[0]), Path(preview_dir) / f"epoch_{epoch:04d}_lr_up.png")
    save_png(tensor_to_numpy(sr[0]), Path(preview_dir) / f"epoch_{epoch:04d}_sr.png")
    save_png(tensor_to_numpy(batch["hr"][0]), Path(preview_dir) / f"epoch_{epoch:04d}_hr.png")
    model.train()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    is_distributed, local_rank, rank, world_size, device = setup_distributed()
    train_cfg = cfg["training"]
    scale = int(cfg["data"]["scale"])
    epochs = args.epochs or int(train_cfg["epochs"])
    batch_size = args.batch_size or int(train_cfg["batch_size"])
    seed_everything(int(train_cfg.get("seed", 42)) + rank)

    dataset = HumanHRDataset(cfg["data"]["human_hr_dir"], scale=scale, limit=args.limit)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if is_distributed else None
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=int(train_cfg.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=int(train_cfg.get("num_workers", 0)) > 0,
    )

    model = SteeringSRUNet(**cfg["model"]).to(device)
    raw_model = model
    ema_decay = float(train_cfg.get("ema_decay", 0.999))
    ema_model = SteeringSRUNet(**cfg["model"]).to(device)
    ema_model.load_state_dict(raw_model.state_dict())
    ema_model.eval()
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    diffusion = GaussianDiffusion(
        timesteps=int(cfg["diffusion"].get("timesteps", 1000)),
        beta_schedule=cfg["diffusion"].get("beta_schedule", "linear"),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg["lr"]))
    checkpoint_dir = ensure_dir(cfg["output"]["checkpoint_dir"])
    log_dir = ensure_dir(cfg["output"]["log_dir"])
    preview_dir = ensure_dir(cfg["output"]["preview_dir"])
    log_path = log_dir / "train_log.csv"
    best_loss = float("inf")
    start_epoch = 1
    sampling_steps = args.sampling_steps or int(cfg["diffusion"].get("sampling_steps", 50))

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        raw_model.load_state_dict(checkpoint["model"])
        if "ema_model" in checkpoint:
            ema_model.load_state_dict(checkpoint["ema_model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_loss = float(checkpoint.get("best_loss", checkpoint.get("loss", best_loss)))

    if is_main_process(rank):
        mode = "a" if args.resume and log_path.exists() else "w"
        with open(log_path, mode, newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["epoch", "loss"])
            if mode == "w":
                writer.writeheader()

    for epoch in range(start_epoch, epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        losses = []
        progress = tqdm(loader, desc=f"epoch {epoch}/{epochs}", disable=not is_main_process(rank))
        for batch in progress:
            hr = batch["hr"].to(device, non_blocking=True)
            lr_up = batch["lr_up"].to(device, non_blocking=True)
            # 多条件状态训练：
            #   ID=2: Human_HR，学习人类高分辨率纹理先验；
            #   ID=1: Human_LR，学习低分辨率桥接状态；
            #   ID=0: Mouse_LR 占位状态，不使用鼠数据，仅用 human LR 退化图像稳定该 embedding。
            # 这样推理阶段三路 orthogonal steering 的三个条件槽位都有训练约束。
            if train_cfg.get("multi_condition_states", True):
                hr = torch.cat([hr, lr_up, lr_up], dim=0)
                lr_up = torch.cat([lr_up, lr_up, lr_up], dim=0)
                batch_size_current = batch["hr"].shape[0]
                condition_tensor = torch.cat(
                    [
                        torch.full((batch_size_current,), 2, device=device, dtype=torch.long),
                        torch.full((batch_size_current,), 1, device=device, dtype=torch.long),
                        torch.full((batch_size_current,), 0, device=device, dtype=torch.long),
                    ],
                    dim=0,
                )
            else:
                condition_tensor = torch.full(
                    (hr.shape[0],),
                    int(train_cfg.get("condition_id", 2)),
                    device=device,
                    dtype=torch.long,
                )
            loss = diffusion.training_loss(
                model,
                hr,
                lr_up,
                condition_tensor,
                float(train_cfg.get("cond_drop_prob", 0.1)),
                float(cfg["diffusion"].get("x0_loss_weight", cfg["diffusion"].get("recon_loss_weight", 0.0))),
                float(cfg["diffusion"].get("gradient_loss_weight", 0.0)),
                float(cfg["diffusion"].get("low_t_prob", 0.0)),
                float(cfg["diffusion"].get("low_t_max_fraction", 0.35)),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_clip = train_cfg.get("grad_clip")
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            optimizer.step()
            update_ema(ema_model, raw_model, ema_decay)
            losses.append(float(loss.item()))
            if is_main_process(rank):
                progress.set_postfix(loss=sum(losses) / len(losses))

        local_loss = sum(losses) / max(1, len(losses))
        epoch_loss = reduce_mean(local_loss, device, is_distributed, world_size)
        if not is_main_process(rank):
            continue
        with open(log_path, "a", newline="", encoding="utf-8") as file:
            csv.DictWriter(file, fieldnames=["epoch", "loss"]).writerow({"epoch": epoch, "loss": epoch_loss})

        payload = {
            "model": raw_model.state_dict(),
            "ema_model": ema_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg,
            "epoch": epoch,
            "loss": epoch_loss,
            "best_loss": min(best_loss, epoch_loss),
            "scale": scale,
        }
        torch.save(payload, checkpoint_dir / f"last_x{scale}.pt")
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(payload, checkpoint_dir / f"best_x{scale}.pt")
        if epoch == 1 or epoch % int(train_cfg.get("save_interval", 20)) == 0:
            torch.save(payload, checkpoint_dir / f"epoch_{epoch:04d}_x{scale}.pt")
            if not args.no_preview:
                save_preview(ema_model, diffusion, next(iter(loader)), epoch, preview_dir, sampling_steps)

    if is_main_process(rank):
        print(f"Training done. best_loss={best_loss:.6f}, checkpoint={checkpoint_dir / f'best_x{scale}.pt'}")
    cleanup_distributed(is_distributed)


if __name__ == "__main__":
    main()
