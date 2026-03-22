import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchvision import transforms
from torchvision.datasets import CIFAR10

# Importaciones de los modelos de difusión y reconstrucción (Basado en el Código 1)
from algos.ddnm import DDNM
from algos.diffpir import DiffPIR
from algos.dps import DPS
from guided_diffusion.script_util import create_model
from utils.SPC_model import SPCModel

# Importaciones del Agente RL y Entorno (Basado en el Código 2)
from agents.agent import ReinforceAgent
from environment.diffusion_env import DiffusionSolverEnv, EpisodeSample
from environment.state_builder import StateBuilder
from solvers.ddnm_solver import DDNMSolver
from solvers.diffpir_solver import DiffPIRSolver
from solvers.dps_solver import DPSSolver
from solvers.solver_library import SolverLibrary


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_backbone(weights_path, image_size, device) -> torch.nn.Module:
    """Carga el modelo base de difusión."""
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Checkpoint not found: {weights_path}")

    checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
    model = create_model(
        image_size=image_size,
        num_channels=64,
        num_res_blocks=3,
        input_channels=3,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def main(opt):
    print("Options:")
    for key, value in vars(opt).items():
        print(f"{key}: {value}")

    device = torch.device(f"cuda:{opt.gpu_id}" if torch.cuda.is_available() and opt.device.startswith("cuda") else "cpu")
    set_seed(opt.seed)

    # ############################## MODELS ############################
    
    # 1. Cargar el modelo de Difusión
    net = build_backbone(opt.weights, opt.image_size, device)

    # 2. Configurar el Entorno y el Agente RL
    diffpir_solver = DiffPIRSolver(model=net, device=device, img_size=opt.image_size, channels=3, steps=opt.diffpir_steps)
    dps_solver = DPSSolver(model=net, device=device, img_size=opt.image_size, channels=3, steps=opt.dps_steps)
    ddnm_solver = DDNMSolver(model=net, device=device, img_size=opt.image_size, channels=3, steps=opt.ddnm_steps)

    solver_library = SolverLibrary(
        diffpir_solver=diffpir_solver,
        dps_solver=dps_solver,
        ddnm_solver=ddnm_solver,
    )

    state_builder = StateBuilder()
    
    # Mapeo de acciones basado en el orden de inicialización de SolverLibrary
    # Ajusta esto si tu SolverLibrary usa un orden distinto.
    action_to_algo = {0: "DiffPIR", 1: "DPS", 2: "DDNM"}

    env = DiffusionSolverEnv(
        solver_library=solver_library,
        state_builder=state_builder,
        max_steps=1,
        device=device,
    )

    # Cargar Agente RL
    agent = ReinforceAgent(
        state_dim=5, # Asegúrate de que coincida con tu StateBuilder
        action_dim=solver_library.action_dim,
        value_coef=0.5,
        entropy_coef=0.01,
    ).to(device)
    
    if os.path.exists(opt.agent_weights):
        agent.load_state_dict(torch.load(opt.agent_weights, map_location=device))
        print(f"Pesos del agente cargados desde {opt.agent_weights}")
    else:
        print("ADVERTENCIA: No se encontraron pesos para el agente. Actuará de forma aleatoria/inicializada.")
    agent.eval()

    # ############################## DATASET ############################

    dataset = CIFAR10(
        root="data",
        train=False,  
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )

    GT = dataset[opt.idx][0].unsqueeze(0).to(device)
    GT = GT * 2 - 1 # Normalizar a [-1, 1]

    # ####################### FORWARD PASS & EPISODE SETUP #######################

    inverse_model = SPCModel(
        im_size=opt.image_size, 
        compression_ratio=opt.sampling_ratio,
        sampling_method=opt.sampling_method,
        device=str(device)
    )
    y = inverse_model.forward_pass(GT)
    x_estimate = inverse_model.transpose_pass(y)

    sample = EpisodeSample(x_true=GT, H=inverse_model, noise_std=opt.measurement_noise_std)
    
    # ####################### AGENT INFERENCE #######################
    
    with torch.no_grad():
        # Obtenemos el estado inicial del entorno
        state = env.reset(episode_sample=sample)
        
        # Le pedimos al agente que tome una decisión
        # NOTA: Dependiendo de tu implementación exacta de ReinforceAgent, puede que retorne logits, 
        # tuplas de (dist, value), o tenga un método get_action. Asumimos un forward estándar que retorna logits/probs.
        policy_output = agent(state) 
        
        # Si policy_output es una tupla (ej. action_probs, state_value), ajusta el índice:
        if isinstance(policy_output, tuple):
            action_probs = policy_output[0]
        else:
            action_probs = policy_output
            
        action = torch.argmax(action_probs, dim=-1).item() # Tomamos la acción más probable (determinista para evaluación)

    chosen_algo = action_to_algo.get(action, "Desconocido")
    print(f"\n---> El Agente RL seleccionó el algoritmo: {chosen_algo} (Acción: {action})\n")

    # ####################### RECONSTRUCTION #######################
    
    # Ejecutamos la lógica original dependiendo de lo que eligió el agente
    if chosen_algo == "DPS":
        diff = DPS(device=device, img_size=opt.image_size, noise_steps=1000, schedule_name="cosine", channels=3, scale=opt.dps_scale, clip_denoised=False)
        reconstruction = diff.sample(model=net, y=y, forward_pass=inverse_model.forward_pass)

    elif chosen_algo == "DDNM":
        diff = DDNM(device=device, img_size=opt.image_size, noise_steps=1000, schedule_name="cosine", channels=3, eta=opt.ddnm_eta)
        reconstruction = diff.sample(model=net, y=y, forward_pass=inverse_model.forward_pass, pseudo_inverse=inverse_model.pseudo_inverse, ground_truth=GT, track_metrics=False)

    elif chosen_algo == "DiffPIR":
        diff = DiffPIR(device=device, img_size=opt.image_size, noise_steps=1000, schedule_name="cosine", channels=3, cg_iters=opt.CG_iters_diffpir, noise_level_img=0.0, iter_num=opt.iter_num, eta=0, zeta=1)
        reconstruction = diff.sample(model=net, y=y, forward_pass=inverse_model.forward_pass, transpose_pass=inverse_model.transpose_pass)

    else:
        raise ValueError(f"Algoritmo {chosen_algo} no soportado.")

    # ####################### METRICS #######################

    SSIM = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    PSNR = PeakSignalNoiseRatio(data_range=1.0).to(device)

    # Normalizar para métricas [0, 1]
    reconstruction_norm = ((reconstruction + 1) / 2).clamp(0, 1)
    GT_norm = ((GT + 1) / 2).clamp(0, 1)

    error_map = (reconstruction_norm - GT_norm).abs().mean(dim=1, keepdim=True)

    ssim_pred = SSIM(reconstruction_norm, GT_norm)
    psnr_pred = PSNR(reconstruction_norm, GT_norm)
    consistency = torch.linalg.norm(inverse_model.forward_pass(reconstruction) - y).item()
    error = torch.linalg.norm(reconstruction_norm - GT_norm).item()

    print(f"SSIM ({chosen_algo}): {ssim_pred:.3f}, PSNR: {psnr_pred:.2f}, Consistency: {consistency:.2f}")

    # ####################### PLOTTING #######################

    fig, ax = plt.subplots(1, 4, figsize=(25, 5))
    ax[0].imshow(GT_norm[0].permute(1, 2, 0).cpu().detach().numpy())
    ax[0].axis("off")
    ax[0].set_title("Ground Truth")

    # Mapeo de la estimación inicial a [0, 1] para visualización
    x_est_vis = ((x_estimate + 1) / 2).clamp(0, 1)
    ax[1].imshow(x_est_vis[0].permute(1, 2, 0).cpu().detach().numpy())
    ax[1].axis("off")
    ax[1].set_title("A^T y")

    ax[2].imshow(reconstruction_norm[0].permute(1, 2, 0).cpu().detach().numpy())
    ax[2].axis("off")
    ax[2].set_title(f"Agente eligió: {chosen_algo}\nSSIM: {ssim_pred:.4f}, PSNR: {psnr_pred:.2f}")

    ax[3].imshow(error_map[0, 0].cpu().detach().numpy(), cmap="hot")
    ax[3].axis("off")
    ax[3].set_title(f"Error Map ({chosen_algo})")
    plt.colorbar(ax[3].images[0], ax=ax[3])
    
    plt.suptitle(f"Inferencia del Agente RL: SPC Reconstruction", fontsize=16, fontweight='bold')
    plt.tight_layout()

    if opt.save_image == "True":
        save_dir = f"results/RL_Agent_Eval/{opt.sampling_method}/{opt.sampling_ratio:.2f}/idx_{opt.idx}"
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(
            f"{save_dir}/reconstruction_{chosen_algo}_psnr_{psnr_pred:.2f}.pdf",
            bbox_inches="tight",
        )

    if opt.plot == "True":
        plt.show()
    else:
        plt.close()

    # ####################### WANDB LOGGING #######################
    if opt.use_wandb == "True":
        wandb.login()
        wandb.init(project=opt.project_name, name=opt.name, config=vars(opt))
        wandb.log({
            "SSIM": ssim_pred.item(),
            "PSNR": psnr_pred.item(),
            "Consistency": consistency,
            "Error": error,
            "Agent_Chosen_Algo": action, # Registramos la acción del agente
        })

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evalúa el Agente RL seleccionando el algoritmo de difusión")
    
    # General
    p.add_argument("--idx", type=int, default=10, help="Índice de la imagen CIFAR10 de prueba")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--save_image", type=str, default="False", choices=["True", "False"])
    p.add_argument("--plot", type=str, default="True", choices=["True", "False"])
    
    # Pesos
    p.add_argument("--weights", type=str, default="weights/e_1000_bs_64_lr_0.0003_seed_2_img_32_schedule_cosine_gpu_0_c_3_si_100/checkpoints/latest.pth.tar")
    p.add_argument("--agent_weights", type=str, default="weights/rl_agent/agent_latest.pth", help="Ruta al checkpoint del Agente RL entrenado")

    # Arquitectura e Imagen
    p.add_argument("--image_size", type=int, default=32)
    p.add_argument("--sampling_ratio", type=float, default=0.5)
    p.add_argument("--sampling_method", type=str, default="gaussian")
    p.add_argument("--measurement_noise_std", type=float, default=0.0)

    # Parámetros de Solvers (para construir el entorno)
    p.add_argument("--diffpir_steps", type=int, default=50)
    p.add_argument("--dps_steps", type=int, default=50)
    p.add_argument("--ddnm_steps", type=int, default=50)
    
    # Parámetros originales de los algoritmos (para la inferencia)
    p.add_argument("--ddnm_eta", type=float, default=1)
    p.add_argument("--dps_scale", type=float, default=0.0125)
    p.add_argument("--CG_iters_diffpir", type=int, default=5)
    p.add_argument("--iter_num", type=int, default=1000)

    # Wandb
    p.add_argument("--use_wandb", type=str, default="False", choices=["True", "False"])
    p.add_argument("--project_name", type=str, default="STSIVA_2026_RL_Eval")
    p.add_argument("--name", type=str, default="eval_agent_one_img")

    main(p.parse_args())