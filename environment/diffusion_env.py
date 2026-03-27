from __future__ import annotations

import argparse
import inspect 
from typing import Any

import torch

from environment.reward import psnr_normalized, ssim_reward
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


class ModelDomainOperator:
    """Operator wrapper that keeps solver/model tensors in [-1, 1]."""

    def __init__(self, operator: Any) -> None:
        self._operator = operator

    @staticmethod
    def model_to_unit(x: torch.Tensor) -> torch.Tensor:
        return torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)

    @staticmethod
    def unit_to_model(x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(2.0 * x - 1.0, -1.0, 1.0)

    def forward_pass(self, x_model: torch.Tensor) -> torch.Tensor:
        x_unit = self.model_to_unit(x_model)
        return self._operator.forward_pass(x_unit)

    def transpose_pass(self, y: torch.Tensor) -> torch.Tensor:
        x_unit = self._operator.transpose_pass(y)
        return self.unit_to_model(torch.clamp(x_unit, 0.0, 1.0))


class DiffusionSolverEnv:
    """Sequential MDP environment for iterative reconstruction."""

    def __init__(
        self,
        solver_library: SolverLibrary,
        state_builder: StateBuilder,
        max_steps: int = 5,
        device: str | torch.device = "cuda",
        psnr_reward_weight: float = 1.0,
        ssim_reward_weight: float = 0.0,
        use_ssim_in_reward: bool = False,
        convergence_tol: float = 1e-8,
        verbose: bool = False,
        args=None,
    ) -> None:
        self.solver_library = solver_library
        self.state_builder = state_builder

        if args is not None:
            self.max_steps = max(1, int(args.max_steps))
            self.device = torch.device(args.device)
            forced_bandit_mode = bool(getattr(args, "bandit_mode", False))
            self.verbose = bool(getattr(args, "verbose", verbose))
        else:
            self.max_steps = max(1, int(max_steps))
            self.device = torch.device(device)
            forced_bandit_mode = False
            self.verbose = bool(verbose)

        self.psnr_reward_weight = float(psnr_reward_weight)
        self.ssim_reward_weight = float(ssim_reward_weight)
        self.use_ssim_in_reward = bool(use_ssim_in_reward)

        self.convergence_tol = float(convergence_tol)

        self.iteration = 0
        self.previous_action: int = 0
        self.current_sample: EpisodeSample | None = None
        self.y: torch.Tensor | None = None
        self.x_current: torch.Tensor | None = None
        self.x_previous: torch.Tensor | None = None
        self.operator_model_domain: ModelDomainOperator | None = None
        self.prev_psnr_norm: float = 0.0
        self.bandit_mode: bool = bool(forced_bandit_mode) or bool(self.solver_library.is_contextual_bandit)
        self._state_builder_accepts_prev_psnr = (
            "previous_psnr" in inspect.signature(self.state_builder.build).parameters
        )

    def _build_state(self) -> torch.Tensor:
        if self.y is None or self.operator_model_domain is None or self.x_current is None:
            raise RuntimeError("Environment state is not initialized.")

        base_kwargs = dict(
            y=self.y.to(self.device),
            H=self.operator_model_domain,
            x_estimate=self.x_current,
            previous_estimate=self.x_previous,
            iteration=self.iteration,
            max_iterations=self.max_steps,
            previous_action=self.previous_action,
            action_count=self.solver_library.action_dim,
        )
        if self._state_builder_accepts_prev_psnr:
            base_kwargs["previous_psnr"] = self.prev_psnr_norm
        return self.state_builder.build(**base_kwargs)


    @staticmethod
    def _model_to_unit(x: torch.Tensor) -> torch.Tensor:
        return torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)

    @staticmethod
    def _unit_to_model(x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(2.0 * x - 1.0, -1.0, 1.0)


    def _build_measurement(self, x_true_model: torch.Tensor, H: Any, noise_std: float) -> torch.Tensor:
        x_true_unit = self._model_to_unit(x_true_model)
        measurement = H.forward_pass(x_true_unit)
        if noise_std > 0.0:
            measurement = measurement + noise_std * torch.randn_like(measurement)
        return measurement
    
    def _compute_consistency_mse(self, x_model: torch.Tensor) -> float:
        if self.current_sample is None or self.y is None:
            return 0.0
        x_unit = self._model_to_unit(x_model)
        residual = self.current_sample.H.forward_pass(x_unit) - self.y
        return float(torch.mean(residual.square()).item())


    def reset(self, sample: EpisodeSample) -> torch.Tensor:
        self.current_sample = sample
        self.iteration = 0
        self.previous_action = 0

        self.operator_model_domain = ModelDomainOperator(sample.H)
        self.y = self._build_measurement(sample.x_true, sample.H, sample.noise_std)

        x0_unit = sample.H.transpose_pass(self.y)
        self.x_current = self._unit_to_model(torch.clamp(x0_unit, 0.0, 1.0)).detach().to(self.device)
        self.x_previous = None

        self.prev_psnr_norm = psnr_normalized(self.x_current, sample.x_true.to(self.device))

        return self._build_state()

    def step(self, action: int) -> tuple[torch.Tensor, float, bool, dict[str, Any]]:
        if self.current_sample is None or self.y is None or self.x_current is None or self.operator_model_domain is None:
            raise RuntimeError("Call reset() before step().")

        x_prev = self.x_current.detach()
        
        x_true = self.current_sample.x_true.to(self.device)
        prev_psnr_norm = psnr_normalized(x_prev, x_true)
        prev_ssim = ssim_reward(x_prev, x_true)

        solver = self.solver_library.get_solver(action)

        solver.set_context(
            x_true=None,
            x_init=x_prev,
            iteration=self.iteration,
            max_iterations=self.max_steps,
            bandit_mode=self.bandit_mode,
        )

        x_next = self.solver_library.apply_solver_step(
            action=action,
            x_k=x_prev,
            y=self.y.to(self.device),
            Phi=self.operator_model_domain,
            ground_truth=x_true,
        )
        x_next = torch.clamp(x_next.detach(), -1.0, 1.0)

        next_psnr_norm = psnr_normalized(x_next, x_true)
        next_ssim = ssim_reward(x_next, x_true)


        delta_psnr_norm = next_psnr_norm - prev_psnr_norm
        delta_ssim = next_ssim - prev_ssim

        if self.use_ssim_in_reward:
            w_psnr = max(0.0, self.psnr_reward_weight)
            w_ssim = max(0.0, self.ssim_reward_weight)
            denom = w_psnr + w_ssim
            if denom <= 0.0:
                w_psnr, w_ssim, denom = 1.0, 0.0, 1.0
            reward = float((w_psnr * delta_psnr_norm + w_ssim * delta_ssim) / denom)
        else:
            reward = float(delta_psnr_norm)
        reward = float(max(-1.0, min(1.0, reward)))




        self.x_previous = x_prev
        self.x_current = x_next
        self.previous_action = int(action)
        self.prev_psnr_norm = next_psnr_norm


        consistency = self._compute_consistency_mse(self.x_current)

        self.iteration += 1
        converged = consistency <= self.convergence_tol
        done = self.bandit_mode or self.iteration >= self.max_steps or converged

        if self.verbose:
            print(
                f"Step {self.iteration}, Action {action}, Reward {reward:.4f}, "
                f"PSNR_norm {prev_psnr_norm:.3f}->{next_psnr_norm:.3f}, "
                f"SSIM {prev_ssim:.3f}->{next_ssim:.3f}, Residual {consistency:.6f}"
            )

        next_state = self._build_state()

        info = {
            "solver": solver.name,
            "reward": reward,
            "psnr_norm": next_psnr_norm,
            "psnr_norm_delta": delta_psnr_norm,
            "ssim": next_ssim,
            "ssim_delta": delta_ssim,
            "psnr_component": next_psnr_norm,
            "consistency_mse": consistency,
            "bandit_mode": self.bandit_mode,
            "converged": converged,
            "iteration": self.iteration,
        }
        return next_state, reward, done, info