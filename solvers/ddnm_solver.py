from __future__ import annotations
from typing import Any
import torch
from algos.ddnm import DDNM


class DDNMSolver:
    """Solver adapter with standardized solve(y, H) interface."""

    name = "DDNM"
    supports_continuation = True

    def __init__(
        self,
        model: torch.nn.Module,
        device: str | torch.device,
        steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        img_size: int = 32,
        schedule_name: str = "cosine",
        channels: int = 1,
        eta: float = 0.85,
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
        self._context = kwargs

    def _blend_factor(self) -> float:
        """Use an iteration-aware blend so multi-step rollouts can reach the solver output."""
        if bool(self._context.get("bandit_mode", False)):
            return 1.0
        max_iterations = max(1, int(self._context.get("max_iterations", 1)))
        iteration = max(0, int(self._context.get("iteration", 0)))
        remaining = max(1, max_iterations - iteration)
        return 1.0 / float(remaining)


    def solve(self, x_k: torch.Tensor | None, y: torch.Tensor, H, ground_truth=None) -> torch.Tensor:
        full_reconstruction = self.solver.sample(
            model=self.model,
            y = y.to(self.device),
            pseudo_inverse=H.transpose_pass,
            forward_pass=H.forward_pass,
            ground_truth=ground_truth,
            track_metrics=False,
        )

        if x_k is None:
            return full_reconstruction

        alpha = self._blend_factor()
        return torch.clamp((1.0 - alpha) * x_k + alpha * full_reconstruction, -1.0, 1.0)


