"""Policy-value network used by the solver-selection RL agent."""

from __future__ import annotations

import torch
from torch import nn


class PolicyNetwork(nn.Module):
    """Two-head MLP policy network.

    Architecture:
    state_dim -> 128 -> 128 -> {action logits, value}
    """

    def __init__(self, state_dim: int, action_dim: int = 3) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(128, action_dim)
        self.value_head = nn.Linear(128, 1)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return action logits and value estimate for each state."""
        features = self.backbone(state)
        logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)
        return logits, value
