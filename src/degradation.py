from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F


def _gaussian_kernel(size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-(grid_x**2 + grid_y**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum().clamp_min(1e-12)
    return kernel.view(1, 1, size, size)


def gaussian_blur(x: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    if sigma <= 0:
        return x
    kernel_size = int(2 * math.ceil(3 * sigma) + 1)
    kernel = _gaussian_kernel(kernel_size, sigma, x.device, x.dtype)
    kernel = kernel.repeat(x.shape[1], 1, 1, 1)
    return F.conv2d(x, kernel, padding=kernel_size // 2, groups=x.shape[1])


def downsample(x: torch.Tensor, scale: int) -> torch.Tensor:
    height, width = x.shape[-2:]
    return F.interpolate(
        x,
        size=(max(1, height // scale), max(1, width // scale)),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )


def upsample(x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    return F.interpolate(x, size=size, mode="bicubic", align_corners=False).clamp(0.0, 1.0)


def degrade_hr_tensor(x: torch.Tensor, scale: int, blur_sigma: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    if blur_sigma is None:
        blur_sigma = 0.6 if scale == 2 else 1.0
    blurred = gaussian_blur(x, blur_sigma)
    lr = downsample(blurred, scale)
    lr_up = upsample(lr, x.shape[-2:])
    return lr, lr_up


def low_frequency_projection(x: torch.Tensor, reference_lr_up: torch.Tensor, scale: int, weight: float) -> torch.Tensor:
    if weight <= 0:
        return x
    _, x_lr_up = degrade_hr_tensor(x.clamp(0.0, 1.0), scale)
    return (x + weight * (reference_lr_up - x_lr_up)).clamp(0.0, 1.0)
