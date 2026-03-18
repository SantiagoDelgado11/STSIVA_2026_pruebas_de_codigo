from __future__ import annotations
import argparse
from operator import add
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
from solvers.ddnm_solver import DDNMSolver
from solvers.diffpir_solver import DiffPIRSolver
from solvers.dps_solver import  DPSSolver
from solvers.solver_library import SolverLibrary
from training.reinforce_trainer import ReinforceTrainer, ReinforceTrainerConfig
from utils.SPC_model import SPCModel
from torchvision.datasets import CIFAR10



def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_backbone(args, device) -> torch.nn.Module:

    if not Path(args.weights).exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.weights}")

    checkpoint = torch.load(args.weights, map_location=device, weights_only=True)
    model = create_model(
        image_size=args.image_size,
        num_channels=args.num_channels,
        num_res_blocks=args.num_res_blocks,
        input_channels=args.input_channels,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model

def train(args):

    set_seed(args.seed)

    use_cuda = torch.cuda.is_available() and str(args.device).startswith("cuda")
    device = torch.device(f"cuda:{args.gpu_id}" if use_cuda else "cpu")

    model = build_backbone(args=args, device=device)

    train_dataset = CIFAR10(root="./data", train=True, download=True, transform=transforms.ToTensor())
    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )


    #SOLVER LIBRARY

    diffpir_solver = DiffPIRSolver(
        model=model,
        device=device,
        config=DiffPIRConfig(
            img_size=args.image_size,
            channels=args.input_channels,
            steps=args.diffpir_steps,
        ),
    )

    dps_solver = DPSSolver(
        model=model,
        device=device,
        config=DPSConfig(
            img_size=args.image_size,
            channels=args.input_channels,
            steps=args.dps_steps,
        ),
    )

    ddnm_solver = DDNMSolver(
        model = model,
        device=device,
        config=DDNMConfig(
            img_size=args.image_size,
            channels=args.input_channels,
            steps=args.ddnm_steps,
        ),
    )

    solver_library = SolverLibrary(
        diffpir_solver=diffpir_solver,
        dps_solver=dps_solver,
        ddnm_solver=ddnm_solver,
    )

# Env + Agent

    state_builder = StateBuilder()

    env = DiffusionSolverEnv(
        solver_library=solver_library,
        state_builder=state_builder,
        max_steps=args.max_env_steps,
        device=device,
    )

    agent = ReinforceAgent(
        state_dim=5,
        action_dim=solver_library.action_dim,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
    ).to(device)

    trainer = ReinforceTrainer(
        agent=agent,
        env=env,
        config=ReinforceTrainerConfig(
            num_episodes=args.num_episodes,
            gamma=args.gamma,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            grad_clip_norm=args.grad_clip_norm,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_every=args.checkpoint_every,
        ),
        device=device,
    )

    data_iter = iter(dataloader)

    def sample_episode():
        nonlocal data_iter

        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        images, _ = batch
        x_true = images[0:1].to(device)
        x_true = x_true * 2.0 - 1.0

        operator = SPCModel(
            im_size=args.image_size,
            compression_ratio=args.sampling_ratio,
            sampling_method=args.sampling_method,
            device=str(device),
        ).to(device)

        return EpisodeSample(
            x_true=x_true,
            H=operator,
            noise_std=args.measurement_noise_std,
        )


    logs = trainer.train(sample_episode)

    rewards = [entry["reward"] for entry in logs]
    print("Training finished.")
    print(f"Episodes: {len(logs)}")
    print(f"Average Reward: {np.mean(rewards):.4f}")
    print(f"Best Reward: {np.max(rewards):.4f}")

def parse_args():
    parser = argparse.ArgumentParser(description="Train RL-based diffusion solver selector.")

    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--data_dir", type=str, default="./data")

    parser.add_argument("--weights", type=str, default="weights/e_1000_bs_64_lr_0.0003_seed_2_img_32_schedule_cosine_gpu_0_c_3_si_100/checkpoints/latest.pth.tar")

    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--input_channels", type=int, default=3)
    parser.add_argument("--num_channels", type=int, default=64)
    parser.add_argument("--num_res_blocks", type=int, default=3)

    parser.add_argument("--num_episodes", type=int, default=1000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--value_coef", type=float, default=0.5)
    parser.add_argument("--entropy_coef", type=float, default=0.01)

    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--checkpoint_dir", type=str, default="weights/rl_agent")
    parser.add_argument("--checkpoint_every", type=int, default=10)

    parser.add_argument("--max_env_steps", type=int, default=1)
    parser.add_argument("--sampling_ratio", type=float, default=0.1)
    parser.add_argument("--sampling_method", type=str, default="gaussian")
    parser.add_argument("--measurement_noise_std", type=float, default=0.0)

    parser.add_argument("--diffpir_steps", type=int, default=50)
    parser.add_argument("--dps_steps", type=int, default=50)
    parser.add_argument("--ddnm_steps", type=int, default=50)

    return parser.parse_args()


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()