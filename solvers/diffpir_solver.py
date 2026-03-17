"""Wrapper for the existing DiffPIR implementation."""

from __future__ import annotations
from typing import Any
import argparse
import torch

from algos.diffpir import DiffPIR


def get_args():
    parser = argparse.ArgumentParser(description="DiffPIR solver configuration")
    parser.add_argument("--noise_steps", type=int, default=1000)
    parser.add_argument("--beta_start", type=float, default=1e-4)
    parser.add_argument("--beta_end", type=float, default=0.02)
    parser.add_argument("--schedule_name", type=str, default="linear")
    parser.add_argument("--cg_iters", type=int, default=20)
    parser.add_argument("--img_size", type=int, default=32)
    parser.add_argument("--channels", type=int, default=3)
    parser.add_argument("--lambda_", type=float, default=1.0)
    parser.add_argument("--noise_level_img", type=float, default=0.0)
    parser.add_argument("--clip_denoised", action="store_true")
    parser.add_argument("--skip_type", type=str, default="quad")
    parser.add_argument("--iter_num", type=int, default=20)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--zeta", type=float, default=1.0)
    return parser.parse_args()


class DiffPIRSolver:
    """Solver adapter with standardized solve(y, H) interface."""

    name = "DiffPIR"

    def __init__(
        self, 
        model: torch.nn.Module, 
        device: str | torch.device, 
        args: argparse.Namespace,
    ) -> None:
        self.model = model
        self.device = torch.device(device)
        self.args = args
        self._context: dict[str, Any] = {}

        self.model.to(self.device)
        self.model.eval()

        self.solver = DiffPIR(
            noise_steps=self.args.noise_steps,
            beta_start=self.args.beta_start,
            beta_end=self.args.beta_end,
            schedule_name=self.args.schedule_name,
            cg_iters=self.args.cg_iters,
            img_size=self.args.img_size,
            channels=self.args.channels,
            lambda_=self.args.lambda_,
            noise_level_img=self.args.noise_level_img,
            clip_denoised=self.args.clip_denoised,
            device=self.device,
            skip_type=self.args.skip_type,
            iter_num=self.args.iter_num,
            eta=self.args.eta,
            zeta=self.args.zeta,
        )

    def set_context(self, **kwargs: Any) -> None:
        
        self._context = kwargs


    def solve(self, y: torch.Tensor, H) -> torch.Tensor:
        y = y.to(self.device)

        with torch.no_grad():
            return self.solver.sample(
                model=self.model,
                y=y,
                forward_pass=H.forward_pass,
                transpose_pass=H.transpose_pass,
                **self._context,  # ahora sí sirve
            )
