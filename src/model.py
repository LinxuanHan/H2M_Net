from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=timesteps.device) / max(half - 1, 1))
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int):
        super().__init__()
        groups1 = min(8, in_ch)
        groups2 = min(8, out_ch)
        self.norm1 = nn.GroupNorm(groups1, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb = nn.Linear(emb_dim, out_ch)
        self.norm2 = nn.GroupNorm(groups2, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb(F.silu(emb)).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class SteeringSRUNet(nn.Module):
    # 条件 ID 映射（复用原 scale embedding 作为联合条件状态指示器）：
    #   0 -> Mouse_LR：小鼠低分辨率目标拓扑条件，用于 zero-shot 推理主分支
    #   1 -> Human_LR：人类低分辨率桥接条件，用于估计跨物种拓扑差异方向
    #   2 -> Human_HR：人类高分辨率纹理条件，用于提取超分高频细节方向
    #   3+ -> 预留：可扩展到 4x 或更多尺度/物种组合状态
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        channel_mults: list[int] | tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
        time_dim: int = 256,
        scale_dim: int = 32,
    ):
        super().__init__()
        self.time_dim = time_dim
        self.scale_embed = nn.Embedding(8, scale_dim)
        emb_dim = time_dim + scale_dim
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))
        self.lr_encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
        )
        self.init = nn.Conv2d(in_channels + base_channels, base_channels, 3, padding=1)

        channels = [base_channels * mult for mult in channel_mults]
        self.downs = nn.ModuleList()
        current = base_channels
        for ch in channels:
            blocks = nn.ModuleList([ResBlock(current if i == 0 else ch, ch, emb_dim) for i in range(num_res_blocks)])
            self.downs.append(nn.ModuleDict({"blocks": blocks, "down": nn.Conv2d(ch, ch, 4, stride=2, padding=1)}))
            current = ch

        self.mid1 = ResBlock(current, current, emb_dim)
        self.mid2 = ResBlock(current, current, emb_dim)

        self.ups = nn.ModuleList()
        for ch in reversed(channels):
            blocks = nn.ModuleList([ResBlock(current + ch if i == 0 else ch, ch, emb_dim) for i in range(num_res_blocks)])
            self.ups.append(nn.ModuleDict({"blocks": blocks, "up": nn.Conv2d(ch, ch, 3, padding=1)}))
            current = ch

        self.out = nn.Sequential(
            nn.GroupNorm(min(8, current), current),
            nn.SiLU(),
            nn.Conv2d(current, in_channels, 3, padding=1),
        )

    def _emb(self, t: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        time_emb = self.time_mlp(sinusoidal_embedding(t, self.time_dim))
        scale = scale.clamp(0, self.scale_embed.num_embeddings - 1)
        return torch.cat([time_emb, self.scale_embed(scale.long())], dim=1)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, lr_up: torch.Tensor | None, scale: torch.Tensor, cond_drop: bool = False) -> torch.Tensor:
        if lr_up is None or cond_drop:
            lr_up = torch.zeros_like(x_t)
        emb = self._emb(t, scale)
        lr_feat = self.lr_encoder(lr_up)
        h = self.init(torch.cat([x_t, lr_feat], dim=1))
        skips = []
        for level in self.downs:
            for block in level["blocks"]:
                h = block(h, emb)
            skips.append(h)
            h = level["down"](h)
        h = self.mid2(self.mid1(h, emb), emb)
        for level in self.ups:
            skip = skips.pop()
            h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = torch.cat([h, skip], dim=1)
            for block in level["blocks"]:
                h = block(h, emb)
            h = level["up"](h)
        return self.out(h)
