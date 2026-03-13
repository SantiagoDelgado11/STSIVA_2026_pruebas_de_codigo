"""Utility dataset for loading test .npy images."""

from __future__ import annotations

import glob
import os
from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset


class TestDataset(Dataset):
    """Dataset of single-channel .npy images.

    Expected input shapes are (H, W), (H, W, 1), or (1, H, W).
    Returns a float tensor with shape (1, H, W), unless a transform is given.
    """

    def __init__(self, root_dir: str, transform: Callable | None = None) -> None:
        self.root_dir = root_dir
        self.transform = transform
        self.files = sorted(glob.glob(os.path.join(root_dir, "*.npy")))

        if not self.files:
            raise RuntimeError(f"No .npy files found in {root_dir}")

    def __len__(self) -> int:
        return len(self.files)

    @staticmethod
    def _to_2d(img: np.ndarray) -> np.ndarray:
        if img.ndim == 2:
            return img
        if img.ndim == 3 and img.shape[-1] == 1:
            return np.squeeze(img, axis=-1)
        if img.ndim == 3 and img.shape[0] == 1:
            return np.squeeze(img, axis=0)
        raise ValueError(f"Expected a single-channel image, got shape {img.shape}")

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = np.load(self.files[idx])
        img = self._to_2d(img).astype(np.float32, copy=False)

        if self.transform is not None:
            transformed = self.transform(img)
            if not isinstance(transformed, torch.Tensor):
                raise TypeError("Transform must return a torch.Tensor")
            return transformed

        return torch.from_numpy(img).unsqueeze(0)


__all__ = ["TestDataset"]
