"""Entry point for training RL-based diffusion solver selection."""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from agents.agent import ReinforceAgent
from environment.diffusion_env import DiffusionSolverEnv, EpisodeSample
from environment.state_builder import StateBuilder
from guided_diffusion.script_util import create_model
from solvers.ddnm_solver import DDNMConfig, DDNMSolver
from solvers.diffpir_solver import DiffPIRConfig, DiffPIRSolver
from solvers.dps_solver import DPSConfig, DPSSolver
from solvers.solver_library import SolverLibrary
from training.reinforce_trainer import ReinforceTrainer, ReinforceTrainerConfig
from utils.SPC_model import SPCModel


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_backbone(config: dict[str, Any], device: torch.device) -> torch.nn.Module:
    """Load pre-trained diffusion model used by all reconstruction solvers."""
    ckpt_path = config["weights"]
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"Backbone checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = create_model(
        image_size=config["image_size"],
        num_channels=config["num_channels"],
        num_res_blocks=config["num_res_blocks"],
        input_channels=config["input_channels"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RL solver-selection agent.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to YAML training configuration.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)  # ACA BA EL ARGPARSER; REVISE TRAIN_DIFF PARA REFERENCIA

    set_seed(int(config.get("seed", 7)))

    use_cuda = torch.cuda.is_available() and str(config.get("device", "cuda")).startswith("cuda")
    device = torch.device(f"cuda:{int(config.get('gpu_id', 0))}" if use_cuda else "cpu")

    model = build_backbone(config=config, device=device)
    dataset = build_dataset(config)  # EL DATAE ES EL CIFAR 10 DE TRAINING
    dataloader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    diffpir_solver = DiffPIRSolver(
        model=model,
        device=device,
        config=DiffPIRConfig(
            img_size=config["image_size"],
            channels=config["input_channels"],
            **config["solver_diffpir"],
        ),
    )
    dps_solver = DPSSolver(
        model=model,
        device=device,
        config=DPSConfig(
            img_size=config["image_size"],
            channels=config["input_channels"],
            **config["solver_dps"],
        ),
    )
    ddnm_solver = DDNMSolver(
        model=model,
        device=device,
        config=DDNMConfig(
            img_size=config["image_size"],
            channels=config["input_channels"],
            **config["solver_ddnm"],
        ),
    )

    solver_library = SolverLibrary(
        diffpir_solver=diffpir_solver,
        dps_solver=dps_solver,
        ddnm_solver=ddnm_solver,
    )

    state_builder = StateBuilder()
    env = DiffusionSolverEnv(
        solver_library=solver_library,
        state_builder=state_builder,
        max_steps=int(config.get("max_env_steps", 1)),
        device=device,
    )

    agent = ReinforceAgent(
        state_dim=5,
        action_dim=solver_library.action_dim,
        value_coef=float(config.get("value_coef", 0.5)),
        entropy_coef=float(config.get("entropy_coef", 0.01)),
    ).to(device)

    trainer = ReinforceTrainer(
        agent=agent,
        env=env,
        config=ReinforceTrainerConfig(
            num_episodes=int(config["num_episodes"]),
            gamma=float(config["gamma"]),
            learning_rate=float(config["learning_rate"]),
            weight_decay=float(config.get("weight_decay", 0.0)),
            grad_clip_norm=float(config.get("grad_clip_norm", 1.0)),
            checkpoint_dir=str(config.get("checkpoint_dir", "weights/rl_solver_selector")),
            checkpoint_every=int(config.get("checkpoint_every", 10)),
        ),
        device=device,
    )

    data_iter = iter(dataloader)

    def sample_episode() -> EpisodeSample:
        nonlocal data_iter
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        x_true = batch[0:1].to(device)
        x_true = x_true * 2.0 - 1.0

        operator = SPCModel(
            im_size=int(config["image_size"]),
            compression_ratio=float(config["sampling_ratio"]),
            sampling_method=str(config["sampling_method"]),
            device=str(device),
        ).to(device)

        return EpisodeSample(
            x_true=x_true,
            H=operator,
            noise_std=float(config.get("measurement_noise_std", 0.0)),
        )

    logs = trainer.train(sample_episode)

    rewards = [entry["reward"] for entry in logs]
    print("Training finished.")
    print(f"Episodes: {len(logs)}")
    print(f"Average reward (PSNR): {np.mean(rewards):.4f}")
    print(f"Best reward (PSNR): {np.max(rewards):.4f}")


if __name__ == "__main__":
    main()
