from __future__ import annotations

import math

import torch


def _to_unit_interval(x: torch.Tensor) -> torch.Tensor:
    """Map tensor to [0, 1] if values look like [-1, 1]."""
    if x.min() < 0.0:
        return ((x + 1.0) / 2.0).clamp(0.0, 1.0)
    return x.clamp(0.0, 1.0)


def psnr_reward(x_hat: torch.Tensor, x_true: torch.Tensor) -> float:
    """Compute PSNR in dB and return it as scalar reward."""
    pred = _to_unit_interval(x_hat)
    target = _to_unit_interval(x_true)

    mse = torch.mean((pred - target) ** 2).item()
    return float(10.0 * math.log10(1.0 / mse))

def psnr_db(x_hat: torch.Tensor, x_true: torch.Tensor) -> float:
    """Compute PSNR in dB (solo para logging / análisis)."""
    pred = _to_unit_interval(x_hat)
    target = _to_unit_interval(x_true)

    mse = torch.mean((pred - target) ** 2).item()
    return float(10.0 * math.log10(1.0 / (mse + 1e-8)))

def ssim_reward(x_hat: torch.Tensor, x_true: torch.Tensor) -> float:

    """Compute SSIM and return it as scalar reward."""
    pred = _to_unit_interval(x_hat)
    target = _to_unit_interval(x_true)

    mean_pred = torch.mean(pred)
    mean_target = torch.mean(target)

    var_pred = torch.var(pred, unbiased=False)
    var_target = torch.var(target, unbiased=False)
    
    covar = torch.mean((pred - mean_pred) * (target - mean_target))

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    numerator = (2 * mean_pred * mean_target + C1) * (2 * covar + C2)
    denominator = (mean_pred ** 2 + mean_target ** 2 + C1) * (var_pred + var_target + C2)

    return float(numerator / denominator)

