"""Wrapper for the existing DiffPIR implementation."""

from __future__ import annotations
from typing import Any
import argparse
import torch

from algos.diffpir import DiffPIR


class DiffPIRSolver:
    """Solver adapter with standardized solve(y, H) interface."""

    name = "DiffPIR"

    def __init__(
        self,
        model: torch.nn.Module,
        device: str | torch.device,
        noise_steps: int,
        beta_start: float,
        beta_end: float,
        schedule_name: str,
        cg_iters: int,
        img_size: int,
        channels: int,
        lambda_: float,
        noise_level_img: float,
        clip_denoised: bool,
        skip_type: str,
        iter_num: int,
        eta: float,
        zeta: float,
    ) -> None:
        self.model = model
        self.device = torch.device(device)
        self._context: dict[str, Any] = {}

        self.model.to(self.device)
        self.model.eval()

        self.solver = DiffPIR(
            noise_steps=noise_steps,
            beta_start=beta_start,
            beta_end=beta_end,
            schedule_name=schedule_name,
            cg_iters=cg_iters,
            img_size=img_size,
            channels=channels,
            lambda_=lambda_,
            noise_level_img=noise_level_img,
            clip_denoised=clip_denoised,
            device=self.device,
            skip_type=skip_type,
            iter_num=iter_num,
            eta=eta,
            zeta=zeta,
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
                **self._context,
            )
