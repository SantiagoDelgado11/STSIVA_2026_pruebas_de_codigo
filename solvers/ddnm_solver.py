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

    def __init__(
            self,
            model: torch.nn.Module,
            device: str | torch.device,
            args = None,
    ) -> None:
        

        if args is None:
            args = get_args()

        self.model = model
        self.device = torch.device(device)
        self.args = args
        self._context: dict[str, Any] = {}

        self.solver = DDNM(
            noise_steps=args.noise_steps,
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            img_size=args.img_size,
            schedule_name=args.schedule_name,
            channels=args.channels,
            eta=args.eta,
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
