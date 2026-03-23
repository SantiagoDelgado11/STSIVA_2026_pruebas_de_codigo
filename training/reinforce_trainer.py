from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import optim

from agents.agent import ReinforceAgent
from environment.diffusion_env import DiffusionSolverEnv, EpisodeSample
from training.rollout import rollout_episode


@dataclass
class ReinforceTrainerConfig:
    """Configuration for REINFORCE training."""

    num_episodes: int = 100
    gamma: float = 1.0
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    grad_clip_norm: float = 0.3
    grad_explosion_threshold: float = 5.0
    checkpoint_dir: str = "weights/rl_solver_selector"
    checkpoint_every: int = 50
    normalize_returns: bool = True
    normalize_advantages: bool = True
    max_abs_advantage: float = 2.5
    reward_scale: float = 1.0
    returns_norm_momentum: float = 0.99
    reward_norm_momentum: float = 0.99
    reward_center: float = 8.0
    reward_temperature: float = 2.0
    critic_loss_type: str = "smooth_l1"
    huber_beta: float = 0.5
    eps: float = 1e-8


class ReinforceTrainer:
    """Trainer that optimizes policy and value heads jointly."""

    def __init__(
        self,
        agent: ReinforceAgent,
        env: DiffusionSolverEnv,
        config: ReinforceTrainerConfig,
        device: str | torch.device,
    ) -> None:
        self.agent = agent
        self.env = env
        self.config = config
        self.device = torch.device(device)

        self.optimizer = optim.Adam(
            self.agent.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self._running_reward_mean = float(config.reward_center)
        self._running_reward_var = float(config.reward_temperature ** 2)
        self._running_return_mean = 0.0
        self._running_return_var = 1.0
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def _save_checkpoint(self, episode: int) -> None:
        ckpt_path = Path(self.config.checkpoint_dir) / f"episode_{episode:05d}.pth"
        torch.save(
            {
                "episode": episode,
                "agent_state": self.agent.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "config": self.config.__dict__,
            },
            ckpt_path,
        )

    def _normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        std = tensor.std(unbiased=False)
        if torch.isnan(std) or std.item() < self.config.eps:
            return tensor - tensor.mean()
        return (tensor - tensor.mean()) / (std + self.config.eps)

    def _normalize_advantages(self, advantages: torch.Tensor) -> torch.Tensor:
        normalized = self._normalize(advantages)
        return torch.clamp(normalized, -self.config.max_abs_advantage, self.config.max_abs_advantage)

    def _normalize_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        if rewards.numel() == 0:
            return rewards

        mean = rewards.mean().item()
        var = rewards.var(unbiased=False).item() if rewards.numel() > 1 else 0.0

        m = self.config.reward_norm_momentum
        self._running_reward_mean = m * self._running_reward_mean + (1.0 - m) * mean
        self._running_reward_var = m * self._running_reward_var + (1.0 - m) * max(var, self.config.eps)

        std = (self._running_reward_var ** 0.5) + self.config.eps
        z = (rewards - self._running_reward_mean) / std
        # Keep reward signal bounded to avoid critic target outliers.
        return torch.tanh(z / max(self.config.reward_temperature, self.config.eps))

    def _normalize_returns_online(self, returns: torch.Tensor) -> torch.Tensor:
        mean = returns.mean().item()
        var = returns.var(unbiased=False).item() if returns.numel() > 1 else 0.0

        m = self.config.returns_norm_momentum
        self._running_return_mean = m * self._running_return_mean + (1.0 - m) * mean
        self._running_return_var = m * self._running_return_var + (1.0 - m) * max(var, self.config.eps)

        denom = (self._running_return_var ** 0.5) + self.config.eps
        normalized = (returns - self._running_return_mean) / denom
        return torch.clamp(normalized, -self.config.max_abs_advantage, self.config.max_abs_advantage)

    def train(self, episode_sampler) -> list[dict[str, float]]:
        self.agent.train()
        logs: list[dict[str, float]] = []

        for episode in range(1, self.config.num_episodes + 1):
            sample: EpisodeSample = episode_sampler()

            trajectory, returns, info = rollout_episode(
                env=self.env,
                agent=self.agent,
                sample=sample,
                gamma=self.config.gamma,
                device=self.device,
            )

            states = torch.stack(trajectory.states).to(self.device)
            actions = torch.tensor(trajectory.actions, dtype=torch.long, device=self.device)

            rewards_tensor = torch.tensor(trajectory.rewards, dtype=torch.float32, device=self.device)
            normalized_rewards = self._normalize_rewards(rewards_tensor) * self.config.reward_scale
            targets = normalized_rewards
            if self.config.normalize_returns:
                targets = self._normalize_returns_online(targets)


            log_probs, entropies, values = self.agent.evaluate_actions(
                states=states,
                actions=actions,
            )

            values = values.squeeze(-1)

            advantages = targets - values.detach()
            if self.config.normalize_advantages:
                advantages = self._normalize_advantages(advantages)

            # REINFORCE objective with standardized advantages before multiplying by log_prob.
            policy_loss = -(log_probs * advantages).mean()
            if self.config.critic_loss_type == "smooth_l1":
                value_loss = F.smooth_l1_loss(values, targets, beta=self.config.huber_beta)
            else:
                value_loss = torch.mean((values - targets) ** 2)
            entropy_bonus = entropies.mean()
            loss = policy_loss + self.agent.value_coef * value_loss - self.agent.entropy_coef * entropy_bonus

            if not torch.isfinite(loss):
                continue

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm_preclip = torch.nn.utils.clip_grad_norm_(
                self.agent.parameters(),
                self.config.grad_clip_norm,
                error_if_nonfinite=False,
            )
            grad_norm_value = float(grad_norm_preclip.item() if isinstance(grad_norm_preclip, torch.Tensor) else grad_norm_preclip)
            grad_exploded = grad_norm_value > self.config.grad_explosion_threshold

            self.optimizer.step()

            episode_log = {
                "episode": float(episode),
                "reward": float(sum(trajectory.rewards)),
                "selected_action": float(trajectory.actions[-1]),
                "selected_solver": float(["DDNM", "DPS", "DiffPIR"].index(info["solver"])),
                "psnr": float(info["psnr"]),
                "loss_total": float(loss.item()),
                "loss_policy": float(policy_loss.item()),
                "loss_value": float(value_loss.item()),
                "entropy": float(entropy_bonus.item()),
                "advantage_mean": float(advantages.mean().item()),
                "returns_mean": float(targets.mean().item()),
                "ratio_mean": 1.0,
                "grad_norm": grad_norm_value,
                "grad_clipped_to": float(self.config.grad_clip_norm),
                "grad_exploded": float(grad_exploded),
                "bandit_mode": float(bool(info.get("bandit_mode", False))),
                "running_return_mean": float(self._running_return_mean),
                "running_return_std": float((self._running_return_var ** 0.5)),
                "running_reward_mean": float(self._running_reward_mean),
                "running_reward_std": float((self._running_reward_var ** 0.5)),

            }
            logs.append(episode_log)

            if episode % 10 == 0 or episode == 1:
                print(
                    f"[Episode {episode:04d}] "
                    f"reward={episode_log['reward']:.3f} "
                    f"solver={info['solver']} "
                    f"psnr={episode_log['psnr']:.3f} "
                    f"loss={episode_log['loss_total']:.4f} "
                    f"grad_norm={episode_log['grad_norm']:.4f}"
                )

            if episode % self.config.checkpoint_every == 0:
                self._save_checkpoint(episode)

        return logs
