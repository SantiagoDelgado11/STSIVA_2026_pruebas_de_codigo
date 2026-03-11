"""RL environment for diffusion solver selection in inverse problems."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from environment.reward import psnr_reward
from environment.state_builder import StateBuilder
from solvers.solver_library import SolverLibrary


@dataclass
class EpisodeSample:
    """Single inverse-problem sample for one episode."""

    x_true: torch.Tensor
    H: Any
    noise_std: float = 0.0


class DiffusionSolverEnv:
    """Environment that maps solver actions to reconstructions and rewards."""

    def __init__(
        self,
        solver_library: SolverLibrary,
        state_builder: StateBuilder,
        max_steps: int = 1,
        device: str | torch.device = "cuda",
    ) -> None:
        self.solver_library = solver_library
        self.state_builder = state_builder
        self.max_steps = max_steps
        self.device = torch.device(device)

        self.iteration = 0
        self.previous_fidelity: float | None = None
        self.current_sample: EpisodeSample | None = None
        self.y: torch.Tensor | None = None
        self.x_init: torch.Tensor | None = None

    def _build_measurement(self, x_true: torch.Tensor, H: Any, noise_std: float) -> torch.Tensor:
        y_clean = H.forward_pass(x_true)
        if noise_std <= 0.0:
            return y_clean
        noise = noise_std * torch.randn_like(y_clean)
        return y_clean + noise

    def reset(self, sample: EpisodeSample) -> torch.Tensor:
        """Reset environment and return initial state."""
        self.current_sample = sample
        self.iteration = 0
        self.previous_fidelity = None

        self.y = self._build_measurement(sample.x_true, sample.H, sample.noise_std)
        if not hasattr(sample.H, "transpose_pass"):
            raise AttributeError("Operator H must implement transpose_pass for state initialization.")

        self.x_init = sample.H.transpose_pass(self.y)

        state = self.state_builder.build(
            y=self.y,
            H=sample.H,
            x_estimate=self.x_init,
            iteration=self.iteration,
            max_iterations=self.max_steps,
            previous_fidelity=self.previous_fidelity,
        )
        self.previous_fidelity = float(state[0].item())
        return state

    def step(self, action: int) -> tuple[torch.Tensor, float, bool, dict[str, Any]]:
        """Execute selected solver and return (next_state, reward, done, info)."""
        if self.current_sample is None or self.y is None:
            raise RuntimeError("Call reset() before step().")

        solver = self.solver_library.get_solver(action)
        solver.set_context(x_true=self.current_sample.x_true)
        x_hat = solver.solve(self.y, self.current_sample.H)

        reward = psnr_reward(x_hat=x_hat, x_true=self.current_sample.x_true)

        self.iteration += 1
        done = self.iteration >= self.max_steps

        next_state = self.state_builder.build(
            y=self.y,
            H=self.current_sample.H,
            x_estimate=x_hat,
            iteration=self.iteration,
            max_iterations=self.max_steps,
            previous_fidelity=self.previous_fidelity,
        )
        self.previous_fidelity = float(next_state[0].item())

        info = {
            "solver": solver.name,
            "psnr": reward,
        }
        return next_state, reward, done, info
