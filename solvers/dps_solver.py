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
    parser.add_argument("--scale", type=float, default=1.0)
    return parser.parse_args()


class DPSSolver:
    """Solver adapter with standardized solve(y, H) interface."""

    name = "DPS"

    def __init__(
        self,
        model: torch.nn.Module,
        device: str | torch.device,
        args=None,
    ) -> None:
        
        if args is None:
            args = get_args()

        self.model = model
        self.device = torch.device(device)
        self.args = args
        self._context: dict[str, Any] = {}

        self.solver = DPS(
            noise_steps=args.noise_steps,
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            img_size=args.img_size,
            device=self.device,
            schedule_name=args.schedule_name,
            channels=args.channels,
            clip_denoised=args.clip_denoised,
            scale=args.scale,
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
