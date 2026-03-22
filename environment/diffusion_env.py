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
    parser.add_argument(
        "--bandit_mode",
        action="store_true",
        help="If set, environment provides zero state and no state information to the agent.",
    )
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
            requested_max_steps = arg.max_steps
            self.device = torch.device(args.device)
            forced_bandit_mode = args.bandit_mode
        else:
            requested_max_steps = max_steps
            self.device = torch.device(device)
            forced_bandit_mode = False

        self.bandit_mode = forced_bandit_mode or (not self.solver_library.supports_continual_actions)
        self.max_steps = 1 if self.bandit_mode else requested_max_steps            

        self.iteration = 0
        self.previous_action: int = 0
        self.current_sample: EpisodeSample | None = None
        self.y: torch.Tensor | None = None
        self.x_current: torch.Tensor | None = None
        self.x_previous: torch.Tensor | None = None

    def _build_measurement(self, x_true: torch.Tensor, H: Any, noise_std: float) -> torch.Tensor:
        _ = noise_std
        measurement = H.forward_pass(x_true)
        if noise_std > 0.0:
            measurement = measurement + noise_std * torch.randn_like(measurement)
        return measurement

    def reset(self, sample: EpisodeSample) -> torch.Tensor:
        """Reset environment and return initial state."""
        self.current_sample = sample
        self.iteration = 0
        self.previous_action = 0

        self.y = self._build_measurement(sample.x_true, sample.H, sample.noise_std)
        if not hasattr(sample.H, "transpose_pass"):
            raise AttributeError("Operator H must implement transpose_pass for state initialization.")

        x0 = sample.H.transpose_pass(self.y)
        self.x_current = x0.detach().to(self.device)
        self.x_previous = None

        state = self.state_builder.build(
            y=self.y,
            H=sample.H,
            x_estimate=self.x_current,
            previous_estimate=self.x_previous,
            iteration=self.iteration,
            max_iterations=self.max_steps,
            previous_action=self.previous_action,
            action_count=self.solver_library.action_dim,    
        )
        return state

    def step(self, action: int) -> tuple[torch.Tensor, float, bool, dict[str, Any]]:
        """Execute selected solver and return (next_state, reward, done, info)."""
        if self.current_sample is None or self.y is None or self.x_current is None:
            raise RuntimeError("Call reset() before step().")

        solver = self.solver_library.get_solver(action)
        solver.set_context(
            x_true=self.current_sample.x_true,
            x_init=self.x_current,
            iteration=self.iteration,
            max_iterations=self.max_steps,
            bandit_mode=self.bandit_mode,
        )   

        x_prev = self.x_current.detach()


        x_hat = self.solver_library.apply_action(
            action=action,
            x_k=x_prev,
            y=self.y,
            Phi=self.current_sample.H,
        )

        self.x_previous = x_prev
        self.x_current = x_hat.detach()
        self.previous_action = action


        reward = psnr_reward(self.x_current, self.current_sample.x_true)
        psnr_real = psnr_db(self.x_current, self.current_sample.x_true)
        ssim = ssim_reward(self.x_current, self.current_sample.x_true)
        residual = self.current_sample.H.forward_pass(self.x_current) - self.y
        consistency = torch.mean(residual.square()).item()


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
            action_count=self.solver_library.action_dim,
        )

        info = {
            "solver": solver.name,
            "psnr": reward,
            "psnr_db": psnr_real,
            "ssim": ssim,
            "consistency_mse": consistency,
            "bandit_mode": self.bandit_mode,

        }
        return next_state, reward, done, info