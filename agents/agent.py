from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.distributions import Categorical
from agents.policy_network import PolicyNetwork


@dataclass
class AgentStep:
    """Container for one interaction step sampled from the policy."""

    action: int
    log_prob: torch.Tensor
    entropy: torch.Tensor
    value: torch.Tensor


class ReinforceAgent(nn.Module):
    """Policy-value agent for discrete solver selection."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 3,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
    ) -> None:
        super().__init__()
        self.network = PolicyNetwork(state_dim=state_dim, action_dim=action_dim)
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

    def select_action(self, state: torch.Tensor) -> AgentStep:
        """Sample action from policy and return statistics needed for REINFORCE."""
        logits, value = self.network(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return AgentStep(
            action=int(action.item()),
            log_prob=dist.log_prob(action),
            entropy=dist.entropy(),
            value=value,
        )

    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return log-probs, entropy and values for a batch of state-action pairs."""
        logits, values = self.network(states)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy, values

    #Make changes here to compute_loss, REINFORCE loss with value and entropy terms

    def compute_loss(
        self,
        log_probs: torch.Tensor,
        values: torch.Tensor,
        returns: torch.Tensor,
        entropies: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute REINFORCE loss with value and entropy terms.

        L = -A * log(pi(a|s)) + c_v * (V(s)-R)^2 - beta * H(pi)
        """
        advantages = returns - values.detach()
        policy_loss = -(advantages * log_probs).mean()
        value_loss = torch.mean((values - returns) ** 2)
        entropy_bonus = entropies.mean()

        total_loss = (
            policy_loss
            + self.value_coef * value_loss
            - self.entropy_coef * entropy_bonus
        )

        metrics = {
            "loss_total": float(total_loss.item()),
            "loss_policy": float(policy_loss.item()),
            "loss_value": float(value_loss.item()),
            "entropy": float(entropy_bonus.item()),
            "advantage_mean": float(advantages.mean().item()),
        }
        return total_loss, metrics
