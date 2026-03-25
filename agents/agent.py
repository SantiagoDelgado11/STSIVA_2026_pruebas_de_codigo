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
        logit_temperature: float = 1.5,
    ) -> None:
        super().__init__()
        self.network = PolicyNetwork(state_dim=state_dim, action_dim=action_dim)
        
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.logit_temperature = max(float(logit_temperature), 1e-3)
    
    def select_action(self, state: torch.Tensor) -> AgentStep:
        logits, value = self.network(state)
        logits = logits / self.logit_temperature
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
        logits = logits / self.logit_temperature
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy, values
