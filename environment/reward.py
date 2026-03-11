"""Reward utilities for reconstruction quality assessment."""

from __future__ import annotations

import math

import torch


def _to_unit_interval(x: torch.Tensor) -> torch.Tensor:
    """Map tensor to [0, 1] if values look like [-1, 1]."""
    if x.min() < 0.0:
        return ((x + 1.0) / 2.0).clamp(0.0, 1.0)
    return x.clamp(0.0, 1.0)


def psnr_reward(x_hat: torch.Tensor, x_true: torch.Tensor, eps: float = 1e-8) -> float:
    """Compute PSNR in dB and return it as scalar reward."""
    pred = _to_unit_interval(x_hat)
    target = _to_unit_interval(x_true)

    mse = torch.mean((pred - target) ** 2).item()
    mse = max(mse, eps)
    return float(10.0 * math.log10(1.0 / mse))
