from __future__ import annotations
from typing import Any
import torch
import argparse
from algos.dps import DPS


class DPSSolver:
    """Solver adapter with standardized solve(y, H) interface."""

    name = "DPS"

    def __init__(
        self,
        model: torch.nn.Module,
        device: str | torch.device,
        steps: int,
        beta_start: float,
        beta_end: float,
        img_size: int,
        schedule_name: str,
        channels: int,
        clip_denoised: bool,
        scale: float,
    ) -> None:

        self.model = model
        self.device = torch.device(device)
        self._context: dict[str, Any] = {}

        self.solver = DPS(
            noise_steps=steps,
            beta_start=beta_start,
            beta_end=beta_end,
            img_size=img_size,
            device=self.device,
            schedule_name=schedule_name,
            channels=channels,
            clip_denoised=clip_denoised,
            scale=scale,
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
