from __future__ import annotations

import argparse
import csv
import os
import tarfile
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("results") / ".matplotlib"))

import matplotlib.pyplot as plt
import torch

from utils.torchvision_compat import patch_torchvision_fake_registration

patch_torchvision_fake_registration()

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchvision import transforms
from torchvision.datasets import CIFAR10

from algos.ddnm import DDNM
from algos.diffpir import DiffPIR
from algos.dps import DPS
from algos.ppo import PPO
from guided_diffusion.script_util import create_model
from utils.SPC_model import SPCModel
from utils.ppo_utils import (
    DEFAULT_DIFFUSION_STEPS,
    DEFAULT_DIFFUSION_WEIGHTS,
    DEFAULT_POLICY_HIDDEN_DIM,
    DEFAULT_PPO_AGENT_WEIGHTS,
    DEFAULT_PPO_ALGO,
    freeze_module,
)
from utils.utils import set_seed


FIELDNAMES = [
    "idx",
    "compression_ratio",
    "method",
    "method_label",
    "psnr",
    "ssim",
    "consistency",
    "l2_error",
]

METHOD_LABELS = {
    "DDNM": "DDNM",
    "DPS": "DPS",
    "DiffPIR": "DiffPIR",
    DEFAULT_PPO_ALGO: "Agent (Proposed)",
}


def build_backbone(weights_path: str, image_size: int, device: torch.device) -> torch.nn.Module:
    if not Path(weights_path).exists():
        raise FileNotFoundError(f"Diffusion checkpoint not found: {weights_path}")

    checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
    model = create_model(
        image_size=image_size,
        num_channels=64,
        num_res_blocks=3,
        input_channels=3,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return freeze_module(model)


def load_cifar10(data_root: str, output_dir: Path, download_data: bool):
    transform = transforms.Compose([transforms.ToTensor()])
    try:
        return CIFAR10(root=data_root, train=False, download=download_data, transform=transform)
    except RuntimeError as exc:
        archive = Path(data_root) / "cifar-10-python.tar.gz"
        if not archive.exists():
            raise RuntimeError(
                f"CIFAR10 was not found in '{data_root}' and archive '{archive}' is missing. "
                "Use --download_data if this machine has internet access."
            ) from exc

        extracted_root = output_dir / "dataset_cache"
        extracted_marker = extracted_root / "cifar-10-batches-py" / "test_batch"
        if not extracted_marker.exists():
            extracted_root.mkdir(parents=True, exist_ok=True)
            print(f"Extracting CIFAR10 archive to {extracted_root}")
            with tarfile.open(archive, "r:gz") as tar:
                tar.extractall(extracted_root, filter="data")

        return CIFAR10(root=str(extracted_root), train=False, download=False, transform=transform)


def read_completed(csv_path: Path) -> set[tuple[int, float, str]]:
    if not csv_path.exists():
        return set()

    completed = set()
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            completed.add((int(row["idx"]), round(float(row["compression_ratio"]), 6), row["method"]))
    return completed


def append_row(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def load_rows(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_summary(csv_path: Path, summary_path: Path) -> None:
    rows = load_rows(csv_path)
    groups: dict[tuple[float, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        key = (float(row["compression_ratio"]), row["method"], row["method_label"])
        groups[key].append(row)

    summary_rows = []
    for (ratio, method, label), group in sorted(groups.items()):
        n = len(group)
        summary_rows.append(
            {
                "compression_ratio": ratio,
                "method": method,
                "method_label": label,
                "n_images": n,
                "mean_psnr": sum(float(r["psnr"]) for r in group) / n,
                "mean_ssim": sum(float(r["ssim"]) for r in group) / n,
                "mean_consistency": sum(float(r["consistency"]) for r in group) / n,
                "mean_l2_error": sum(float(r["l2_error"]) for r in group) / n,
            }
        )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    csv_summary_path = summary_path.with_suffix(".csv")
    with csv_summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()) if summary_rows else [])
        if summary_rows:
            writer.writeheader()
            writer.writerows(summary_rows)

    lines = [
        "# Compression Ratio Evaluation",
        "",
        "| Compression ratio | Method | N | PSNR | SSIM | Consistency | L2 error |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| "
            f"{row['compression_ratio']:.2f} | "
            f"{row['method_label']} | "
            f"{row['n_images']} | "
            f"{row['mean_psnr']:.4f} | "
            f"{row['mean_ssim']:.4f} | "
            f"{row['mean_consistency']:.4f} | "
            f"{row['mean_l2_error']:.4f} |"
        )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_method(opt, method: str, model, operator, y, gt, device: torch.device) -> tuple[torch.Tensor, dict | None]:
    if method == "DPS":
        diff = DPS(
            device=device,
            img_size=opt.image_size,
            noise_steps=opt.diffusion_steps,
            schedule_name="cosine",
            channels=3,
            scale=opt.dps_scale,
            clip_denoised=False,
        )
        return diff.sample(model=model, y=y, forward_pass=operator.forward_pass), None

    if method == "DDNM":
        diff = DDNM(
            device=device,
            img_size=opt.image_size,
            noise_steps=opt.diffusion_steps,
            schedule_name="cosine",
            channels=3,
            eta=opt.ddnm_eta,
        )
        return (
            diff.sample(
                model=model,
                y=y,
                forward_pass=operator.forward_pass,
                pseudo_inverse=operator.pseudo_inverse,
                ground_truth=gt,
                track_metrics=False,
            ),
            None,
        )

    if method == "DiffPIR":
        diff = DiffPIR(
            device=device,
            img_size=opt.image_size,
            noise_steps=opt.diffusion_steps,
            schedule_name="cosine",
            channels=3,
            cg_iters=opt.CG_iters_diffpir,
            noise_level_img=opt.noise_level_img,
            iter_num=opt.iter_num or opt.diffusion_steps,
            eta=opt.diffpir_eta,
            zeta=opt.diffpir_zeta,
            lambda_=opt.diffpir_lambda,
            skip_type=opt.skip_type,
        )
        return (
            diff.sample(
                model=model,
                y=y,
                forward_pass=operator.forward_pass,
                transpose_pass=operator.transpose_pass,
            ),
            None,
        )

    if method == DEFAULT_PPO_ALGO:
        diff = PPO(
            model=model,
            agent_weights=opt.agent_weights,
            device=device,
            img_size=opt.image_size,
            noise_steps=opt.diffusion_steps,
            channels=3,
            hidden_dim=opt.policy_hidden_dim,
            ddnm_eta=opt.ddnm_eta,
            dps_scale=opt.dps_scale,
            cg_iters=opt.CG_iters_diffpir,
            diffpir_lambda=opt.diffpir_lambda,
            noise_level_img=opt.noise_level_img,
            diffpir_eta=opt.diffpir_eta,
            diffpir_zeta=opt.diffpir_zeta,
        )
        return diff.sample(y=y, operator=operator)

    raise ValueError(f"Unknown method: {method}")


def evaluate_reconstruction(reconstruction, gt, operator, y, psnr_metric, ssim_metric) -> dict:
    reconstruction = ((reconstruction + 1) / 2).clamp(0.0, 1.0)
    gt = ((gt + 1) / 2).clamp(0.0, 1.0)
    return {
        "psnr": psnr_metric(reconstruction, gt).item(),
        "ssim": ssim_metric(reconstruction, gt).item(),
        "consistency": torch.linalg.norm(operator.forward_pass(reconstruction * 2 - 1) - y).item(),
        "l2_error": torch.linalg.norm(reconstruction - gt).item(),
    }


def save_figure(fig_path: Path, gt, x_estimate, reconstructions: dict[str, torch.Tensor], metrics: dict[str, dict]) -> None:
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    gt_vis = ((gt + 1) / 2).clamp(0.0, 1.0)
    x_estimate_vis = ((x_estimate + 1) / 2).clamp(0.0, 1.0)

    methods = list(reconstructions.keys())
    fig, axes = plt.subplots(len(methods), 4, figsize=(16, 4 * len(methods)))
    if len(methods) == 1:
        axes = axes[None, :]

    for row, method in enumerate(methods):
        reconstruction = ((reconstructions[method] + 1) / 2).clamp(0.0, 1.0)
        error_map = (reconstruction - gt_vis).abs().mean(dim=1, keepdim=True)

        panels = [
            (gt_vis[0].permute(1, 2, 0).cpu().numpy(), "Ground Truth", None),
            (x_estimate_vis[0].permute(1, 2, 0).cpu().numpy(), "A^T y", None),
            (
                reconstruction[0].permute(1, 2, 0).cpu().numpy(),
                f"{METHOD_LABELS[method]}\nPSNR {metrics[method]['psnr']:.2f}, SSIM {metrics[method]['ssim']:.3f}",
                None,
            ),
            (error_map[0, 0].cpu().numpy(), "Error map", "hot"),
        ]

        for col, (image, title, cmap) in enumerate(panels):
            axes[row, col].imshow(image, cmap=cmap)
            axes[row, col].set_title(title)
            axes[row, col].axis("off")
            if cmap:
                plt.colorbar(axes[row, col].images[0], ax=axes[row, col], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(fig_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


def main(opt) -> None:
    device = torch.device(f"cuda:{opt.gpu_id}" if torch.cuda.is_available() and opt.device.startswith("cuda") else "cpu")
    set_seed(opt.seed)

    output_dir = Path(opt.output_dir)
    csv_path = output_dir / "per_image_metrics.csv"
    summary_path = output_dir / "summary.md"
    completed = read_completed(csv_path) if opt.resume else set()

    if DEFAULT_PPO_ALGO in opt.methods and not Path(opt.agent_weights).exists():
        raise FileNotFoundError(
            f"Agent checkpoint not found: {opt.agent_weights}. "
            "Train/copy the proposed-agent checkpoint there, or run without --methods PPO."
        )

    model = build_backbone(opt.weights, opt.image_size, device)
    dataset = load_cifar10(opt.data_root, output_dir, opt.download_data)

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    indices = list(range(opt.start_idx, opt.start_idx + opt.num_images))
    figure_indices = set(opt.figure_indices)

    for ratio in opt.ratios:
        operator = SPCModel(im_size=opt.image_size, compression_ratio=ratio).to(device)
        for idx in indices:
            gt = dataset[idx][0].unsqueeze(0).to(device) * 2 - 1
            y = operator.forward_pass(gt)
            x_estimate = operator.transpose_pass(y)
            figure_recons = {}
            figure_metrics = {}

            for method in opt.methods:
                key = (idx, round(float(ratio), 6), method)
                if key in completed:
                    print(f"Skipping completed idx={idx} ratio={ratio:.2f} method={method}")
                    continue

                print(f"Running idx={idx} ratio={ratio:.2f} method={method}")
                set_seed(opt.seed + idx)
                reconstruction, action_summary = run_method(opt, method, model, operator, y, gt, device)
                metric_values = evaluate_reconstruction(reconstruction, gt, operator, y, psnr_metric, ssim_metric)

                append_row(
                    csv_path,
                    {
                        "idx": idx,
                        "compression_ratio": f"{ratio:.6f}",
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        **{k: f"{v:.8f}" for k, v in metric_values.items()},
                    },
                )
                completed.add(key)

                if idx in figure_indices:
                    figure_recons[method] = reconstruction.detach().cpu()
                    figure_metrics[method] = metric_values

                if action_summary:
                    print(f"PPO solver counts: {action_summary['counts']}")

                del reconstruction
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if idx in figure_indices and figure_recons:
                save_figure(
                    output_dir / "figures" / f"idx_{idx}_ratio_{ratio:.2f}.png",
                    gt.detach().cpu(),
                    x_estimate.detach().cpu(),
                    figure_recons,
                    figure_metrics,
                )

            write_summary(csv_path, summary_path)

    write_summary(csv_path, summary_path)
    print(f"Done. Metrics: {csv_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=DEFAULT_DIFFUSION_WEIGHTS)
    parser.add_argument("--agent_weights", type=str, default=DEFAULT_PPO_AGENT_WEIGHTS)
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--download_data", action="store_true")
    parser.add_argument("--output_dir", type=str, default="results/compression_ratio_eval")
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--num_images", type=int, default=50)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--figure_indices", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--ratios", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.3, 0.4])
    parser.add_argument("--methods", type=str, nargs="+", default=["DDNM", "DPS", "DiffPIR", DEFAULT_PPO_ALGO])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="Skip rows already present in per_image_metrics.csv")

    parser.add_argument("--diffusion_steps", type=int, default=DEFAULT_DIFFUSION_STEPS)
    parser.add_argument("--ddnm_eta", type=float, default=1.0)
    parser.add_argument("--dps_scale", type=float, default=0.0125)
    parser.add_argument("--skip_type", type=str, default="uniform", choices=["uniform", "quad"])
    parser.add_argument("--iter_num", type=int, default=0, help="DiffPIR iterations. 0 uses diffusion_steps.")
    parser.add_argument("--CG_iters_diffpir", type=int, default=5)
    parser.add_argument("--noise_level_img", type=float, default=0.0)
    parser.add_argument("--diffpir_lambda", type=float, default=1.0)
    parser.add_argument("--diffpir_eta", type=float, default=0.0)
    parser.add_argument("--diffpir_zeta", type=float, default=1.0)
    parser.add_argument("--policy_hidden_dim", type=int, default=DEFAULT_POLICY_HIDDEN_DIM)

    main(parser.parse_args())
