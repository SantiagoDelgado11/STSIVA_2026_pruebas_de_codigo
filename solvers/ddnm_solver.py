"""Wrapper for the existing DDNM implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from algos.ddnm import DDNM


@dataclass
class DDNMConfig:
    """Configuration used to instantiate DDNM."""

    noise_steps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    img_size: int = 256
    schedule_name: str = "cosine"
    channels: int = 1
    eta: float = 1.0
    extra_kwargs: dict[str, Any] = field(default_factory=dict)


class DDNMSolver:
    """Solver adapter with standardized solve(y, H) interface."""

    name = "DDNM"

    def __init__(self, model: torch.nn.Module, device: str | torch.device, config: DDNMConfig) -> None:
        self.model = model
        self.device = str(device)
        self.config = config
        self._context: dict[str, Any] = {}

        self.solver = DDNM(
            noise_steps=config.noise_steps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            img_size=config.img_size,
            device=self.device,
            schedule_name=config.schedule_name,
            channels=config.channels,
            eta=config.eta,
            **config.extra_kwargs,
        )

    def set_context(self, **kwargs: Any) -> None:
        """Set optional context (x_true used by existing DDNM signature)."""
        self._context = kwargs

    def solve(self, y: torch.Tensor, H) -> torch.Tensor:
        """Run DDNM and return the reconstruction."""
        x_true = self._context.get("x_true")
        if x_true is None:
            raise ValueError("DDNMSolver requires x_true in context before calling solve().")

        return self.solver.sample(
            model=self.model,
            y=y,
            pseudo_inverse=H.transpose_pass,
            forward_pass=H.forward_pass,
            ground_truth=x_true,
            track_metrics=False,
        )
