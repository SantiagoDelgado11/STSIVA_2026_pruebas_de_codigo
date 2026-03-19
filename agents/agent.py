from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical
from agents.policy_network import PolicyNetwork

class AgentStep:
    """Container for one interaction step sampled from the policy."""

    def __init__(
        self,
        action: int,
        log_prob: torch.Tensor,
        entropy: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        self.action = action
        self.log_prob = log_prob
        self.entropy = entropy
        self.value = value

class ReinforceAgent(nn.Module):        

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 3,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        eps_clip: float = 0.2
    ) -> None:
        super().__init__()
        self.network = PolicyNetwork(state_dim=state_dim, action_dim=action_dim)
        
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.eps_clip = eps_clip
    
    def select_action(self, state: torch.Tensor) -> AgentStep:
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

        logits, values = self.network(states)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy, values

    def compute_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        values: torch.Tensor,
        returns: torch.Tensor,
        entropies: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:

        advantages = returns - values.detach()

        ratio = torch.exp(log_probs - old_log_probs.detach())

        surr1 = ratio * advantages
        surr2 = torch.clamp(
            ratio,
            1.0 - self.eps_clip,
            1.0 + self.eps_clip,
        ) * advantages

        policy_loss = -torch.min(surr1, surr2).mean()

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
            "ratio_mean": float(ratio.mean().item()),
        }
        return total_loss, metrics     
