"""REINFORCE training loop for diffusion solver selection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import optim

from agents.agent import ReinforceAgent
from environment.diffusion_env import DiffusionSolverEnv, EpisodeSample
from training.rollout import rollout_episode


@dataclass
class ReinforceTrainerConfig:
    """Configuration for REINFORCE training."""

    num_episodes: int = 100
    gamma: float = 1.0
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0
    checkpoint_dir: str = "weights/rl_solver_selector"
    checkpoint_every: int = 50


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

    def train(self, episode_sampler) -> list[dict[str, float]]:
        """Train for configured episodes.

        episode_sampler must return EpisodeSample instances.
        """
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

            log_probs = torch.stack(trajectory.log_probs)
            values = torch.stack(trajectory.values)
            entropies = torch.stack(trajectory.entropies)

            loss, metrics = self.agent.compute_loss(
                log_probs=log_probs,
                values=values,
                returns=returns,
                entropies=entropies,
            )

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.config.grad_clip_norm)
            self.optimizer.step()

            episode_log = {
                "episode": float(episode),
                "reward": float(sum(trajectory.rewards)),
                "selected_action": float(trajectory.actions[-1]),
                "selected_solver": float(["DDNM", "DPS", "DiffPIR"].index(info["solver"])),
                "psnr": float(info["psnr"]),
                **metrics,
            }
            logs.append(episode_log)

            if episode % 10 == 0 or episode == 1:
                print(
                    f"[Episode {episode:04d}] "
                    f"reward={episode_log['reward']:.3f} "
                    f"solver={info['solver']} "
                    f"psnr={episode_log['psnr']:.3f} "
                    f"loss={metrics['loss_total']:.4f}"
                )

            if episode % self.config.checkpoint_every == 0:
                self._save_checkpoint(episode)

        return logs
