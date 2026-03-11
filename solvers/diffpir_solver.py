"""Wrapper for the existing DiffPIR implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from algos.diffpir import DiffPIR


@dataclass
class DiffPIRConfig:
    """Configuration used to instantiate DiffPIR."""

    noise_steps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    schedule_name: str = "linear"
    cg_iters: int = 20
    img_size: int = 256
    channels: int = 1
    lambda_: float = 1.0
    noise_level_img: float = 0.0
    clip_denoised: bool = False
    skip_type: str = "quad"
    iter_num: int = 20
    eta: float = 0.0
    zeta: float = 1.0
    extra_kwargs: dict[str, Any] = field(default_factory=dict)


class DiffPIRSolver:
    """Solver adapter with standardized solve(y, H) interface."""

    name = "DiffPIR"

    def __init__(self, model: torch.nn.Module, device: str | torch.device, config: DiffPIRConfig) -> None:
        self.model = model
        self.device = str(device)
        self.config = config
        self._context: dict[str, Any] = {}

        self.solver = DiffPIR(
            noise_steps=config.noise_steps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            schedule_name=config.schedule_name,
            cg_iters=config.cg_iters,
            img_size=config.img_size,
            channels=config.channels,
            lambda_=config.lambda_,
            noise_level_img=config.noise_level_img,
            clip_denoised=config.clip_denoised,
            device=self.device,
            skip_type=config.skip_type,
            iter_num=config.iter_num,
            eta=config.eta,
            zeta=config.zeta,
            **config.extra_kwargs,
        )

    def set_context(self, **kwargs: Any) -> None:
        """Store optional contextual information for compatibility."""
        self._context = kwargs

    def solve(self, y: torch.Tensor, H) -> torch.Tensor:
        """Run DiffPIR and return the reconstruction."""
        return self.solver.sample(
            model=self.model,
            y=y,
            forward_pass=H.forward_pass,
            transpose_pass=H.transpose_pass,
        )
