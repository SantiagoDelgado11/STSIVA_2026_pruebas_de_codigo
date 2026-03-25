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
    parser.add_argument(
        "--bandit_mode",
        action="store_true",
        help="If set, force contextual-bandit operation (single step per episode).",
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


class ModelDomainOperator:
    """Operator wrapper that keeps solver/model tensors in [-1, 1].

    - forward_pass receives model-domain tensor and maps to [0, 1] before the physical operator.
    - transpose_pass receives measurement-domain tensor and maps reconstructed image back to [-1, 1].
    """

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
    """Environment that maps solver actions to reconstructions and rewards."""

    def __init__(
        self,
        solver_library: SolverLibrary,
        state_builder: StateBuilder,
        max_steps: int = 5,
        device: str | torch.device = "cuda",
        psnr_reward_weight: float = 0.7,
        ssim_reward_weight: float = 0.3,
        use_ssim_in_reward: bool = True,
        psnr_norm_min_db: float = 10.0,
        psnr_norm_max_db: float = 40.0,
        psnr_norm_target_range: str = "minus_one_to_one",
        args=None,
    ) -> None:
        self.solver_library = solver_library
        self.state_builder = state_builder

        if args is not None:
            requested_max_steps = int(args.max_steps)
            self.device = torch.device(args.device)
            forced_bandit_mode = bool(args.bandit_mode)
        else:
            requested_max_steps = int(max_steps)
            self.device = torch.device(device)
            forced_bandit_mode = False

        # If solvers cannot continue from x_k, the interaction is a contextual bandit.
        self.bandit_mode = forced_bandit_mode or self.solver_library.is_contextual_bandit
        self.max_steps = 1 if self.bandit_mode else max(1, requested_max_steps)

        self.psnr_reward_weight = float(psnr_reward_weight)
        self.ssim_reward_weight = float(ssim_reward_weight)
        self.use_ssim_in_reward = bool(use_ssim_in_reward)

        self.psnr_norm_min_db = float(psnr_norm_min_db)
        self.psnr_norm_max_db = float(psnr_norm_max_db)
        self.psnr_norm_target_range = str(psnr_norm_target_range)
        if self.psnr_norm_max_db <= self.psnr_norm_min_db:
            raise ValueError("psnr_norm_max_db must be greater than psnr_norm_min_db.")
        if self.psnr_norm_target_range not in {"zero_to_one", "minus_one_to_one"}:
            raise ValueError("psnr_norm_target_range must be 'zero_to_one' or 'minus_one_to_one'.")

        self.iteration = 0
        self.previous_action: int = 0
        self.current_sample: EpisodeSample | None = None
        self.y: torch.Tensor | None = None
        self.x_current: torch.Tensor | None = None
        self.x_previous: torch.Tensor | None = None
        self.operator_model_domain: ModelDomainOperator | None = None

    @staticmethod
    def _model_to_unit(x: torch.Tensor) -> torch.Tensor:
        return torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)

    @staticmethod
    def _unit_to_model(x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(2.0 * x - 1.0, -1.0, 1.0)

    def _psnr_db_unit(self, x_hat_unit: torch.Tensor, x_true_unit: torch.Tensor) -> float:
        mse = torch.mean((x_hat_unit - x_true_unit) ** 2).item()
        return float(10.0 * math.log10(1.0 / (mse + 1e-8)))

    def _ssim_unit(self, x_hat_unit: torch.Tensor, x_true_unit: torch.Tensor) -> float:
        mean_pred = torch.mean(x_hat_unit)
        mean_target = torch.mean(x_true_unit)

        var_pred = torch.var(x_hat_unit, unbiased=False)
        var_target = torch.var(x_true_unit, unbiased=False)
        covar = torch.mean((x_hat_unit - mean_pred) * (x_true_unit - mean_target))

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        numerator = (2.0 * mean_pred * mean_target + c1) * (2.0 * covar + c2)
        denominator = (mean_pred ** 2 + mean_target ** 2 + c1) * (var_pred + var_target + c2)
        return float(numerator / (denominator + 1e-8))

    def _normalize_psnr(
        self,
        psnr_db: float,
    ) -> float:
        """Min-max normalized PSNR with explicit saturation bounds.

        1) Saturate to [psnr_norm_min_db, psnr_norm_max_db]
        2) Min-max map to [0, 1]
        3) Optionally map to [-1, 1]
        """
        if not math.isfinite(psnr_db):
            clipped = self.psnr_norm_min_db
        else:
            clipped = max(min(psnr_db, self.psnr_norm_max_db), self.psnr_norm_min_db)

        unit = (clipped - self.psnr_norm_min_db) / (self.psnr_norm_max_db - self.psnr_norm_min_db)
        if self.psnr_norm_target_range == "zero_to_one":
            return float(unit)
        return float((2.0 * unit) - 1.0)

    def _build_measurement(self, x_true_model: torch.Tensor, H: Any, noise_std: float) -> torch.Tensor:
        x_true_unit = self._model_to_unit(x_true_model)
        measurement = H.forward_pass(x_true_unit)
        if noise_std > 0.0:
            measurement = measurement + noise_std * torch.randn_like(measurement)
        return measurement

    def reset(self, sample: EpisodeSample) -> torch.Tensor:
        """Reset environment and return initial state."""
        self.current_sample = sample
        self.iteration = 0
        self.previous_action = 0

        self.operator_model_domain = ModelDomainOperator(sample.H)
        self.y = self._build_measurement(sample.x_true, sample.H, sample.noise_std)

        # Initialize from H^T y in unit domain and map to model domain for the solvers.
        x0_unit = sample.H.transpose_pass(self.y)
        self.x_current = self._unit_to_model(torch.clamp(x0_unit, 0.0, 1.0)).detach().to(self.device)
        self.x_previous = None

        state = self.state_builder.build(
            y=self.y.to(self.device),
            H=self.operator_model_domain,
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
        if self.current_sample is None or self.y is None or self.x_current is None or self.operator_model_domain is None:
            raise RuntimeError("Call reset() before step().")

        solver = self.solver_library.get_solver(action)
        solver.set_context(
            x_true=None,
            x_init=self.x_current,
            iteration=self.iteration,
            max_iterations=self.max_steps,
            bandit_mode=self.bandit_mode,
        )

        x_prev = self.x_current.detach()
        x_hat = self.solver_library.apply_action(
            action=action,
            x_k=x_prev,
            y=self.y.to(self.device),
            Phi=self.operator_model_domain,
            ground_truth=self.ground_truth
        )

        self.x_previous = x_prev
        self.x_current = torch.clamp(x_hat.detach(), -1.0, 1.0)
        self.previous_action = int(action)

        x_hat_unit = self._model_to_unit(self.x_current)
        x_true_unit = self._model_to_unit(self.current_sample.x_true.to(self.device))

        psnr_real = self._psnr_db_unit(x_hat_unit, x_true_unit)
        ssim = self._ssim_unit(x_hat_unit, x_true_unit)

        psnr_component = self._normalize_psnr(psnr_real)
        ssim_component = max(min((2.0 * ssim) - 1.0, 1.0), -1.0)

        if self.use_ssim_in_reward:
            reward = (self.psnr_reward_weight * psnr_component) + (self.ssim_reward_weight * ssim_component)
        else:
            reward = psnr_component

        # Consistency is computed against physical-domain measurements [0, 1].
        residual = self.current_sample.H.forward_pass(x_hat_unit) - self.y
        consistency = torch.mean(residual.square()).item()

        self.iteration += 1
        done = self.iteration >= self.max_steps

        next_state = self.state_builder.build(
            y=self.y.to(self.device),
            H=self.operator_model_domain,
            x_estimate=self.x_current,
            previous_estimate=self.x_previous,
            iteration=self.iteration,
            max_iterations=self.max_steps,
            previous_action=self.previous_action,
            action_count=self.solver_library.action_dim,
        )

        info = {
            "solver": solver.name,
            "reward": reward,
            "psnr": psnr_real,
            "psnr_component": psnr_component,
            "ssim": ssim,
            "ssim_component": ssim_component,
            "consistency_mse": consistency,
            "bandit_mode": self.bandit_mode,
        }
        return next_state, reward, done, info
