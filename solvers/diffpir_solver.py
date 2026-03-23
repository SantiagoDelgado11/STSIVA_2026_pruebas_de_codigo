from __future__ import annotations
from typing import Any
import torch

from algos.diffpir import DiffPIR


class DiffPIRSolver:
    """Solver adapter with continuation-aware interface."""

    name = "DiffPIR"
    supports_continuation = False

    def __init__(
        self,
        model: torch.nn.Module,
        device: str | torch.device,
        steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        schedule_name: str = "cosine",
        cg_iters: int = 40,
        img_size: int = 32,
        channels: int = 1,
        lambda_: float = 1.0,
        noise_level_img: float = 1.0,
        clip_denoised: bool = False,
        skip_type: str = "quad",
        iter_num: int = 20,
        eta: float = 0.0,
        zeta: float = 1.0,
    ) -> None:
        self.model = model
        self.device = torch.device(device)
        self._context: dict[str, Any] = {}

        self.model.to(self.device)
        self.model.eval()

        self.solver = DiffPIR(
            noise_steps=steps,
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

    def solve(self, x_k: torch.Tensor | None, y: torch.Tensor, H) -> torch.Tensor:
        _ = x_k

        with torch.no_grad():
            return self.solver.sample(
                model=self.model,
                y=y.to(self.device),
                forward_pass=H.forward_pass,
                transpose_pass=H.transpose_pass,
            )
