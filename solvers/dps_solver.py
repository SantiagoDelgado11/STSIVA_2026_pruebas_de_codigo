from __future__ import annotations
from typing import Any
import torch
import argparse
from algos.dps import DPS


def get_args():
    parser = argparse.ArgumentParser(description="DPS solver configuration")
    parser.add_argument("--noise_steps", type=int, default=1000)
    parser.add_argument("--beta_start", type=float, default=1e-4)
    parser.add_argument("--beta_end", type=float, default=0.02)
    parser.add_argument("--img_size", type=int, default=32)
    parser.add_argument("--schedule_name", type=str, default="cosine")
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--clip_denoised", action="store_true")
    parser.add_argument("--scale", type=float, default=0.0125)
    return parser.parse_args()


class DPSConfig:
    """Configuration used to instantiate DPS."""

    noise_steps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    img_size: int = 32
    schedule_name: str = "cosine"
    channels: int = 3
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
