from __future__ import annotations

import torch


class StateBuilder:
    """Build bounded state vectors for iterative solver control."""

    def __init__(self, args=None) -> None:
        self.eps = float(getattr(args, "eps", 1e-8))
        self.state_clip = float(getattr(args, "state_clip", 1.0))
        self.log_scale = float(getattr(args, "log_scale", 6.0))
        self.zscore_momentum = float(getattr(args, "zscore_momentum", 0.95))
        self.state_dim = 11

        self._running_mean: list[float] = [0.0] * self.state_dim
        self._running_var: list[float] = [1.0] * self.state_dim

    def _signed_log1p(self, value: torch.Tensor) -> torch.Tensor:
        return torch.sign(value) * torch.log1p(value.abs())

    def _bounded_log_feature(self, value: torch.Tensor) -> torch.Tensor:
        scaled = self._signed_log1p(value) / self.log_scale
        return torch.clamp(scaled, -1.0, 1.0)

    def _standardize_feature(self, index: int, value: torch.Tensor) -> torch.Tensor:
        scalar = float(value.detach().item())

        mean_prev = self._running_mean[index]
        var_prev = self._running_var[index]
        momentum = self.zscore_momentum

        mean_new = momentum * mean_prev + (1.0 - momentum) * scalar
        centered = scalar - mean_new
        var_new = momentum * var_prev + (1.0 - momentum) * (centered * centered)

        self._running_mean[index] = mean_new
        self._running_var[index] = max(var_new, self.eps)

        z = centered / (self._running_var[index] ** 0.5 + self.eps)
        return torch.tensor(float(torch.tanh(torch.tensor(z / 3.0))), dtype=torch.float32, device=value.device)

    def _normalize_action(self, previous_action: int, action_count: int, device: torch.device) -> torch.Tensor:
        if action_count <= 1:
            return torch.tensor(0.0, device=device)
        centered = (2.0 * float(previous_action) / float(action_count - 1)) - 1.0
        return torch.tensor(centered, dtype=torch.float32, device=device)

    def _normalize_psnr(self, psnr_db: float, device: torch.device) -> torch.Tensor:
        return torch.tensor(float(psnr_db), dtype=torch.float32, device=device).clamp(-1.0, 1.0)

    def build(
        self,
        y: torch.Tensor,
        H,
        x_estimate: torch.Tensor,
        previous_estimate: torch.Tensor | None,
        iteration: int,
        max_iterations: int,
        previous_action: int,
        action_count: int,
        previous_psnr: float,
    ) -> torch.Tensor:
        """Build a bounded state vector for a single environment step."""
        device = x_estimate.device
        eps = self.eps

        residual = H.forward_pass(x_estimate) - y
        fidelity = torch.mean(residual.square())

        gradient = H.transpose_pass(residual)
        grad_norm = torch.linalg.norm(gradient)

        x_energy = torch.mean(x_estimate.square())
        measurement_energy = torch.mean(y.square())
        spatial_variance = torch.var(x_estimate, unbiased=False)

        if previous_estimate is None:
            convergence_ratio = torch.tensor(0.0, device=device)
            cosine_alignment = torch.tensor(1.0, device=device)
        else:
            delta = x_estimate - previous_estimate
            convergence_ratio = torch.linalg.norm(delta) / (torch.linalg.norm(x_estimate) + eps)

            flat_x = x_estimate.reshape(-1)
            flat_prev = previous_estimate.reshape(-1)
            cosine_alignment = torch.nn.functional.cosine_similarity(
                flat_x.unsqueeze(0),
                flat_prev.unsqueeze(0),
                dim=1,
                eps=eps,
            ).squeeze(0)

        residual_to_signal = fidelity / (measurement_energy + eps)
        normalized_iteration = torch.tensor(
            2.0 * float(iteration) / max(float(max_iterations), 1.0) - 1.0,
            dtype=torch.float32,
            device=device,
        )
        previous_action_feature = self._normalize_action(previous_action, action_count, device)
        prev_psnr_feature = self._normalize_psnr(previous_psnr, device)

        raw_features = [
            self._bounded_log_feature(fidelity),
            self._bounded_log_feature(grad_norm),
            self._bounded_log_feature(measurement_energy),
            self._bounded_log_feature(x_energy),
            self._bounded_log_feature(spatial_variance),
            torch.tanh(convergence_ratio),
            torch.clamp(cosine_alignment, -1.0, 1.0),
            self._bounded_log_feature(residual_to_signal),
            torch.clamp(normalized_iteration, -1.0, 1.0),
            torch.clamp(previous_action_feature, -1.0, 1.0),
            torch.clamp(prev_psnr_feature, -1.0, 1.0),
        ]

        normalized_features = [
            self._standardize_feature(i, feature.to(device=device, dtype=torch.float32))
            for i, feature in enumerate(raw_features)
        ]

        state = torch.stack(normalized_features).float()
        return torch.clamp(state, -self.state_clip, self.state_clip)
