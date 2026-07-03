from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .degradation import degrade_hr_tensor, upsample
from .utils import list_images, load_image


def _to_tensor(array) -> torch.Tensor:
    tensor = torch.from_numpy(array).float()
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 3:
        tensor = tensor[:1]
    return tensor.clamp(0.0, 1.0)


class HumanHRDataset(Dataset):
    def __init__(self, hr_dir: str, scale: int, limit: int | None = None):
        self.hr_paths = list_images(hr_dir)
        if limit is not None:
            self.hr_paths = self.hr_paths[:limit]
        if not self.hr_paths:
            raise FileNotFoundError(f"No images found in {hr_dir}")
        self.scale = int(scale)

    def __len__(self) -> int:
        return len(self.hr_paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.hr_paths[index]
        hr = _to_tensor(load_image(path))
        _, lr_up = degrade_hr_tensor(hr.unsqueeze(0), self.scale)
        return {
            "name": path.stem,
            "hr": hr,
            "lr_up": lr_up.squeeze(0),
            "scale": torch.tensor(self.scale, dtype=torch.long),
        }

class MouseInferenceDataset(Dataset):
    def __init__(
        self,
        lr_dir: str,
        scale: int,
        hr_dir: str | None = None,
        mask_dir: str | None = None,
        limit: int | None = None,
    ):
        self.lr_paths = list_images(lr_dir)
        if limit is not None:
            self.lr_paths = self.lr_paths[:limit]
        if not self.lr_paths:
            raise FileNotFoundError(f"No LR images found in {lr_dir}")
        self.scale = int(scale)
        self.hr_by_stem = self._index_optional(hr_dir)
        self.mask_by_stem = self._index_optional(mask_dir)

    @staticmethod
    def _index_optional(folder: str | None) -> dict[str, Path]:
        if not folder:
            return {}
        paths = list_images(folder)
        return {path.stem: path for path in paths}

    def __len__(self) -> int:
        return len(self.lr_paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        lr_path = self.lr_paths[index]
        lr = _to_tensor(load_image(lr_path))
        hr_path = self.hr_by_stem.get(lr_path.stem)
        hr = _to_tensor(load_image(hr_path)) if hr_path else None
        target_size = tuple(hr.shape[-2:]) if hr is not None else (lr.shape[-2] * self.scale, lr.shape[-1] * self.scale)
        lr_up = upsample(lr.unsqueeze(0), target_size).squeeze(0)
        mask_path = self.mask_by_stem.get(lr_path.stem)
        mask = _to_tensor(load_image(mask_path)) if mask_path else None
        return {
            "name": lr_path.stem,
            "lr": lr,
            "lr_up": lr_up,
            "hr": hr if hr is not None else torch.empty(0),
            "mask": mask if mask is not None else torch.empty(0),
            "has_hr": hr is not None,
            "has_mask": mask is not None,
            "scale": torch.tensor(self.scale, dtype=torch.long),
        }
