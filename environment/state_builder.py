"""State feature construction based on lightweight solver statistics."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class StateBuilderConfig:
    """Configuration for state vector generation."""

    eps: float = 1e-8


class StateBuilder:
    """Build compact state vectors from inverse-problem statistics.

    Feature order:
    1. data fidelity value
    2. gradient norm
    3. convergence ratio
    4. normalized iteration index
    5. measurement norm
    6. estimated noise level
    """

    def __init__(self, config: StateBuilderConfig | None = None) -> None:
        self.config = config or StateBuilderConfig()

    def build(
        self,
        y: torch.Tensor,
        H,
        x_estimate: torch.Tensor,
        iteration: int,
        max_iterations: int,
        previous_fidelity: float | None = None,
    ) -> torch.Tensor:
        """Build a 6-D feature vector for a single environment step."""
        eps = self.config.eps

        residual = H.forward_pass(x_estimate) - y
        fidelity = torch.linalg.norm(residual) / (torch.linalg.norm(y) + eps)

        if hasattr(H, "transpose_pass"):
            gradient = H.transpose_pass(residual)
            grad_norm = torch.linalg.norm(gradient) / (torch.linalg.norm(H.transpose_pass(y)) + eps)
        else:
            grad_norm = torch.linalg.norm(residual) / (torch.linalg.norm(y) + eps)

        if previous_fidelity is None:
            convergence_ratio = torch.tensor(1.0, device=y.device)
        else:
            convergence_ratio = fidelity / (torch.tensor(previous_fidelity, device=y.device) + eps)

        normalized_iteration = torch.tensor(
            float(iteration) / float(max(max_iterations, 1)),
            device=y.device,
        )
        measurement_norm = torch.linalg.norm(y) / torch.sqrt(torch.tensor(float(y.numel()), device=y.device))

        centered_y = y - y.mean()
        noise_estimate = torch.median(torch.abs(centered_y)) / 0.6745

        state = torch.stack(
            [
                fidelity,
                grad_norm,
                convergence_ratio,
                normalized_iteration,
                measurement_norm,
                noise_estimate,
            ]
        ).float()
        return state
