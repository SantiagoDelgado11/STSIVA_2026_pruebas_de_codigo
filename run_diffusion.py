import os


batch_size = 170
image_size = 32
seed = 0
gpu_id = 0
save_image = "False"
plot = "False"
idx_list = [i for i in range(0, batch_size)]
wandb_id = "b879bf20f3c31bfcf13289e363f4d3394f7d7671"
use_wandb = "True"
device = "cuda"
weights = "weights/e_1000_bs_64_lr_0.0003_seed_2_img_32_schedule_cosine_gpu_0_c_3_si_100/checkpoints/latest.pth.tar"  # Diffusion weights

sampling_ratio = 1 / 12
sampling_method = "uniform"
algo = "DDNM"


# DDNM parameters
ddnm_eta = 1.0

# DPS parameters
dps_scale = 0.0125

# DiffPIR parameters
CG_iters_diffpir = 100
noise_level_img = 0.0


project_name = "NullDiff"  # f"{algo}_{sampling_method}_{180 * sampling_ratio // 1}"


for idx in idx_list:
    name = f"{idx}_{algo}_{sampling_method}_{180 * sampling_ratio // 1}"
    os.system(
        f"python run_one_img.py --idx {idx} --weights {weights} --batch_size {batch_size} --image_size {image_size} --seed {seed} --device {device} --gpu_id {gpu_id} --save_image {save_image} --plot {plot} --sampling_ratio {sampling_ratio} --sampling_method {sampling_method} --algo {algo} --dps_scale {dps_scale} --CG_iter {CG_iter} --CE_iter {CE_iter} --mu {mu} --rho {rho} --CG_iters_diffpir {CG_iters_diffpir} --noise_level_img {noise_level_img} --fista_iter {fista_iter} --fista_step {fista_step} --denoiser_strength {denoiser_strength} --use_wandb {use_wandb} --wandb_id {wandb_id} --project_name {project_name} --name {name}"
    )
