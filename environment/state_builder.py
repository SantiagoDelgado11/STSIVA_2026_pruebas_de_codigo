"""State feature construction based on lightweight solver statistics."""

from __future__ import annotations
import argparse
import torch


def get_args():
    parser = argparse.ArgumentParser(description="StateBuilder configuration")
    parser.add_argument("--eps", type=float, default=1e-8)
    return parser.parse_known_args()[0]


class StateBuilder:
    """Build compact state vectors from inverse-problem statistics.

    Feature order:
    1. data fidelity g(x_k) = ||Phi x_k - y||^2
    2. gradient norm ||Phi^T(Phi x_k - y)||
    3. convergence ratio ||x_k - x_{k-1}|| / ||x_k||
    4. normalized iteration index k / K
    5. previous action a_{k-1}
    """

    def __init__(self, args=None) -> None:
        if args is None:
            args = get_args()

        self.eps = args.eps

    def build(
        self,
        y: torch.Tensor,
        H,
        x_estimate: torch.Tensor,
        previous_estimate: torch.Tensor | None,
        iteration: int,
        max_iterations: int,
        previous_action: int,
    ) -> torch.Tensor:
        """Build a 5-D feature vector for a single environment step."""
        eps = self.eps

        residual = H.forward_pass(x_estimate) - y
        fidelity = torch.linalg.norm(residual) ** 2

        gradient = H.transpose_pass(residual)
        grad_norm = torch.linalg.norm(gradient)

        if previous_estimate is None:
            convergence_ratio = torch.tensor(0.0, device=y.device)
        else:
            convergence_ratio = torch.linalg.norm(x_estimate - previous_estimate) / (torch.linalg.norm(x_estimate) + eps)

        normalized_iteration = torch.tensor(
            float(iteration) / float(max_iterations),
            device=y.device,
        )
        previous_action_feature = torch.tensor(float(previous_action), device=y.device)

        state = torch.stack(
            [
                fidelity,
                grad_norm,
                convergence_ratio,
                normalized_iteration,
                previous_action_feature,
            ]
        ).float()
        return state
