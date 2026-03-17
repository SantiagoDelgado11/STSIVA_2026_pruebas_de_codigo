"""Wrapper for the existing DPS implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from algos.dps import DPS


@dataclass
class DPSConfig:
    """Configuration used to instantiate DPS."""

    noise_steps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    img_size: int = 32
    schedule_name: str = "cosine"
    channels: int = 1
    clip_denoised: bool = False
    scale: float = 0.0125
    extra_kwargs: dict[str, Any] = field(default_factory=dict)


class DPSSolver:
    """Solver adapter with standardized solve(y, H) interface."""

    name = "DPS"

    def __init__(self, model: torch.nn.Module, device: str | torch.device, config: DPSConfig) -> None:
        self.model = model
        self.device = str(device)
        self.config = config
        self._context: dict[str, Any] = {}

        self.solver = DPS(
            noise_steps=config.noise_steps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            img_size=config.img_size,
            device=self.device,
            schedule_name=config.schedule_name,
            channels=config.channels,
            clip_denoised=config.clip_denoised,
            scale=config.scale,
            **config.extra_kwargs,
        )

    def set_context(self, **kwargs: Any) -> None:
        """Store optional contextual information for compatibility."""
        self._context = kwargs

    def solve(self, y: torch.Tensor, H) -> torch.Tensor:
        """Run DPS and return the reconstruction."""
        return self.solver.sample(
            model=self.model,
            y=y,
            forward_pass=H.forward_pass,
        )
