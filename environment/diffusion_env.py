from __future__ import annotations

import argparse
import math
from typing import Any

import torch

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
        args=None,
    ) -> None:
        self.solver_library = solver_library
        self.state_builder = state_builder

        if args is not None:
            self.max_steps = max(1, int(args.max_steps))
            self.device = torch.device(args.device)
            forced_bandit_mode = bool(args.bandit_mode)
        else:
            self.max_steps = max(1, int(max_steps))
            self.device = torch.device(device)
            

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
        self.prev_psnr: float = 0.0

    @staticmethod
    def _model_to_unit(x: torch.Tensor) -> torch.Tensor:
        return torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)

    @staticmethod
    def _unit_to_model(x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(2.0 * x - 1.0, -1.0, 1.0)

    def _psnr_db_unit(self, x_hat_unit: torch.Tensor, x_true_unit: torch.Tensor) -> float:
        mse = torch.mean((x_hat_unit - x_true_unit) ** 2).item()
        return float(10.0 * math.log10(1.0 / (mse + 1e-8)))

    def _normalize_psnr(self, psnr_db: float) -> float:
        if not math.isfinite(psnr_db):
            return -1.0    
        scaled = (psnr_db - 10.0) / 20.0
        return float(math.tanh(scaled))

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

        x_true_unit = self._model_to_unit(sample.x_true.to(self.device))
        self.prev_psnr = self._psnr_db_unit(self._model_to_unit(self.x_current), x_true_unit)

        state = self.state_builder.build(
            y=self.y.to(self.device),
            H=self.operator_model_domain,
            x_estimate=self.x_current,
            previous_estimate=self.x_previous,
            iteration=self.iteration,
            max_iterations=self.max_steps,
            previous_action=self.previous_action,
            action_count=self.solver_library.action_dim,
            previous_psnr=self.prev_psnr,
        )
        return state

    def step(self, action: int) -> tuple[torch.Tensor, float, bool, dict[str, Any]]:
        if self.current_sample is None or self.y is None or self.x_current is None or self.operator_model_domain is None:
            raise RuntimeError("Call reset() before step().")

        x_prev = self.x_current.detach()
        
        x_true_unit = self._model_to_unit(self.current_sample.x_true.to(self.device))

        prev_psnr = self._psnr_db_unit(self._model_to_unit(x_prev), x_true_unit)

        solver = self.solver_library.get_solver(action)

        solver.set_context(
            x_true=None,
            x_init=x_prev,
            iteration=self.iteration,
            max_iterations=self.max_steps,
            bandit_mode=False,
        )

        x_next = self.solver_library.apply_solver_step(
            action=action,
            x_k=x_prev,
            y=self.y.to(self.device),
            Phi=self.operator_model_domain,
            ground_truth=self.current_sample.x_true.to(self.device),
        )
        x_next = torch.clamp(x_next.detach(), -1.0, 1.0)

        next_psnr = self._psnr_db_unit(self._model_to_unit(x_next), x_true_unit)
        reward = self.psnr_reward_weight * (next_psnr - prev_psnr)


        self.x_previous = x_prev
        self.x_current = x_next
        self.previous_action = int(action)
        self.prev_psnr = next_psnr

        consistency = self._compute_consistency_mse(self.x_current)

        self.iteration += 1
        converged = consistency <= self.convergence_tol
        done = self.iteration >= self.max_steps or converged

        print(
            f"Step {self.iteration}, Action {action}, Reward {reward:.4f}, "
            f"PSNR {prev_psnr:.3f}->{next_psnr:.3f}, Residual {consistency:.6f}"
        )

        next_state = self.state_builder.build(
            y=self.y.to(self.device),
            H=self.operator_model_domain,
            x_estimate=self.x_current,
            previous_estimate=self.x_previous,
            iteration=self.iteration,
            max_iterations=self.max_steps,
            previous_action=self.previous_action,
            action_count=self.solver_library.action_dim,
            previous_psnr=self.prev_psnr,
        )

        info = {
            "solver": solver.name,
            "reward": reward,
            "psnr": next_psnr,
            "psnr_delta": (next_psnr - prev_psnr),
            "psnr_component": self._normalize_psnr(next_psnr),
            "consistency_mse": consistency,
            "bandit_mode": False,
            "converged": converged,
            "iteration": self.iteration,
        }
        return next_state, reward, done, info
