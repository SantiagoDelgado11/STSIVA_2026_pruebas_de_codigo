from __future__ import annotations
import argparse
from typing import Any
import torch

from environment.reward import psnr_reward, ssim_reward, psnr_db
from environment.state_builder import StateBuilder
from solvers.solver_library import SolverLibrary


def get_args():
    parser = argparse.ArgumentParser(description="Diffusion RL Environment")
    parser.add_argument("--max_steps", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


class EpisodeSample:
    """Single inverse-problem sample for one episode."""

    def __init__(
        self,
        x_true: torch.Tensor,
        H: Any,
        noise_std: float = 0.0,
    ) -> None:
        self.x_true = x_true
        self.H = H
        self.noise_std = noise_std



class DiffusionSolverEnv:
    """Environment that maps solver actions to reconstructions and rewards."""

    def __init__(
        self,
        solver_library: SolverLibrary,
        state_builder: StateBuilder,
        max_steps: int = 5,
        device: str | torch.device = "cuda",
        args=None,
    ) -> None:

        self.solver_library = solver_library
        self.state_builder = state_builder
        
        if args is not None:
            self.max_steps = args.max_steps
            self.device =  torch.device(args.device)
        else:
            self.max_steps = max_steps
            self.device = torch.device(device)         

        self.iteration = 0
        self.previous_action: int = 0
        self.current_sample: EpisodeSample | None = None
        self.y: torch.Tensor | None = None
        self.x_current: torch.Tensor | None = None
        self.x_previous: torch.Tensor | None = None

    def _build_measurement(self, x_true: torch.Tensor, H: Any, noise_std: float) -> torch.Tensor:
        _ = noise_std
        return H.forward_pass(x_true)

    def reset(self, sample: EpisodeSample) -> torch.Tensor:
        """Reset environment and return initial state."""
        self.current_sample = sample
        self.iteration = 0
        self.previous_action = 0

        self.y = self._build_measurement(sample.x_true, sample.H, sample.noise_std)
        if not hasattr(sample.H, "transpose_pass"):
            raise AttributeError("Operator H must implement transpose_pass for state initialization.")

        self.x_current = sample.H.transpose_pass(self.y)
        self.x_previous = self.x_current.clone()

        state = self.state_builder.build(
            y=self.y,
            H=sample.H,
            x_estimate=self.x_current,
            previous_estimate=self.x_previous,
            iteration=self.iteration,
            max_iterations=self.max_steps,
            previous_action=self.previous_action,
        )
        return state

    def step(self, action: int) -> tuple[torch.Tensor, float, bool, dict[str, Any]]:
        """Execute selected solver and return (next_state, reward, done, info)."""
        if self.current_sample is None or self.y is None or self.x_current is None:
            raise RuntimeError("Call reset() before step().")

        solver = self.solver_library.get_solver(action)
        solver.set_context(x_true=self.current_sample.x_true)
        
        x_prev = self.x_current

        x_hat = self.solver_library.apply_action(
            action=action,
            x_k=x_prev,
            y=self.y,
            Phi=self.current_sample.H,
        )

        self.x_previous = x_prev
        self.x_current = x_hat
        self.previous_action = action

        reward = psnr_reward(x_hat, self.current_sample.x_true)
        psnr_real = psnr_db(x_hat, self.current_sample.x_true)
        ssim = ssim_reward(x_hat, self.current_sample.x_true)

        self.iteration += 1
        done = self.iteration >= self.max_steps

        next_state = self.state_builder.build(
            y=self.y,
            H=self.current_sample.H,
            x_estimate=self.x_current,
            previous_estimate=self.x_previous,
            iteration=self.iteration,
            max_iterations=self.max_steps,
            previous_action=self.previous_action,
        )

        info = {
            "solver": solver.name,
            "psnr": reward,
            "psnr_db": psnr_real,
            "ssim": ssim,
        }
        return next_state, reward, done, info