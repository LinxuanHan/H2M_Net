from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .utils import ensure_dir


def _to_uint8(image):
    image = np.squeeze(np.asarray(image, dtype=np.float32))
    return np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)


def save_comparison(lr_up, sr, hr, path: str | Path, title: str = ""):
    ensure_dir(Path(path).parent)
    panels = [("LR up", lr_up), ("SR", sr)]
    if hr is not None and np.size(hr):
        panels.append(("HR", hr))
        panels.append(("|SR-HR|", np.abs(np.squeeze(sr) - np.squeeze(hr))))
    images = [Image.fromarray(_to_uint8(image)).convert("RGB") for _, image in panels]
    width, height = images[0].size
    label_h = 22
    canvas = Image.new("RGB", (width * len(images), height + label_h), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, ((name, _), image) in enumerate(zip(panels, images)):
        canvas.paste(image.resize((width, height)), (idx * width, label_h))
        draw.text((idx * width + 4, 4), name, fill=(0, 0, 0))
    canvas.save(path)
