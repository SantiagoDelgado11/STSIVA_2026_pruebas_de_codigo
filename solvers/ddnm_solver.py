from __future__ import annotations
from typing import Any
import torch
import argparse 
from algos.ddnm import DDNM

def get_args():
    parser = argparse.ArgumentParser(description="DDNM solver configuration")
    parser.add_argument("--noise_steps", type=int, default=1000)
    parser.add_argument("--beta_start", type=float, default=1e-4)
    parser.add_argument("--beta_end", type=float, default=0.02)
    parser.add_argument("--img_size", type=int, default=32)
    parser.add_argument("--schedule_name", type=str, default="cosine")
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--eta", type=float, default=0.85)
    return parser.parse_args()

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
