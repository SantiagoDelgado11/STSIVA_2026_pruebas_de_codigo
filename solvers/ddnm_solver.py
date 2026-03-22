from __future__ import annotations
from typing import Any
import torch
import argparse
from algos.ddnm import DDNM


class DDNMSolver:
    """Solver adapter with standardized solve(y, H) interface."""

    name = "DDNM"

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
        eta: float,
    ) -> None:

        self.model = model
        self.device = torch.device(device)
        self._context: dict[str, Any] = {}

        self.solver = DDNM(
            noise_steps=steps,
            beta_start=beta_start,
            beta_end=beta_end,
            img_size=img_size,
            schedule_name=schedule_name,
            channels=channels,
            eta=eta,
        )

    def set_context(self, **kwargs: Any) -> None:
        """Set optional context (x_true used by existing DDNM signature)."""
        self._context = kwargs

    def solve(self, y: torch.Tensor, H) -> torch.Tensor:
        """Run DDNM and return the reconstruction."""
        x_true = self._context.get("x_true", None)

        return self.solver.sample(
            model=self.model,
            y=y,
            pseudo_inverse=H.transpose_pass,
            forward_pass=H.forward_pass,
            ground_truth=x_true,
            track_metrics=False,
        )
