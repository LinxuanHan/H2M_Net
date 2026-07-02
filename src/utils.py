import json
import random
import warnings
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".npy", ".nii", ".gz"}


def load_config(path):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def list_image_files(root):
    root = Path(root)
    files = []
    for path in root.rglob("*"):
        if path.is_file() and (path.suffix.lower() in IMAGE_SUFFIXES or path.name.lower().endswith(".nii.gz")):
            files.append(path)
    return sorted(files)


def list_images(root):
    return list_image_files(root)


def normalize01(array):
    array = np.asarray(array, dtype=np.float32)
    finite = np.isfinite(array)
    if not finite.any():
        return np.zeros_like(array, dtype=np.float32)
    valid = array[finite]
    min_value = float(valid.min())
    max_value = float(valid.max())
    if max_value <= min_value:
        return np.zeros_like(array, dtype=np.float32)
    array = (array - min_value) / (max_value - min_value)
    array[~finite] = 0
    return np.clip(array, 0.0, 1.0).astype(np.float32)


def load_medical_image(path):
    path = Path(path)
    suffix = path.suffix.lower()
    meta = {"path": str(path), "affine": None}
    if path.name.lower().endswith(".nii.gz") or suffix == ".nii":
        try:
            import nibabel as nib
        except ImportError as error:
            raise ImportError("nibabel is required to read NIfTI files.") from error
        image = nib.load(str(path))
        meta["affine"] = image.affine
        array = np.asarray(image.get_fdata(), dtype=np.float32)
    elif suffix == ".npy":
        array = np.load(path).astype(np.float32)
    else:
        array = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    return normalize01(array), meta


def load_image(path):
    array, _ = load_medical_image(path)
    if array.ndim > 2:
        array = np.asarray(array)
        if array.shape[-1] <= 4:
            array = array[..., 0]
        else:
            array = array.take(indices=array.shape[-1] // 2, axis=-1)
    return normalize01(array)


def save_png(array, path):
    ensure_dir(Path(path).parent)
    array = np.squeeze(np.asarray(array, dtype=np.float32))
    image = Image.fromarray(np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8))
    image.save(path)


def save_npy(array, path):
    ensure_dir(Path(path).parent)
    np.save(path, np.asarray(array, dtype=np.float32))


def save_nifti(array, path, affine=None):
    try:
        import nibabel as nib
    except ImportError:
        warnings.warn("nibabel is not installed; skip NIfTI saving.")
        return
    ensure_dir(Path(path).parent)
    if affine is None:
        affine = np.eye(4)
    nib.save(nib.Nifti1Image(np.asarray(array, dtype=np.float32), affine), str(path))


def tensor_to_numpy(tensor):
    return tensor.detach().squeeze().float().cpu().numpy()


def save_json(data, path):
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def warn(message):
    warnings.warn(message, stacklevel=2)
