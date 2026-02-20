"""
Train a Conditional 3D Latent Diffusion Model.

The model learns to predict future fraction latents conditioned on the first
fraction latent via channel concatenation.

Framework : PyTorch + MONAI DiffusionModelUNet
Scheduler : DDPM (cosine beta schedule, predicts clean sample x_0)
Hardware  : Multi-GPU via DataParallel (optimised for 4x A100 40 GB)

Usage:
    python train_diffusion.py [--data-dir DIR] [--save-dir DIR] [--epochs N]
"""

import os
import glob
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from monai.networks.nets import DiffusionModelUNet
from monai.utils import set_determinism
from diffusers import DDPMScheduler

from dataset import LatentPairsDataset

# ---------------------------------------------------------------------------
# Default hyper-parameters
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    "data_dir": "/net/tscratch/people/plgpiotreksl/maisi_vae_latent",
    "save_dir": "/net/tscratch/people/plgpiotreksl/models/conditional_diffusion_unet",
    "batch_size": 16,
    "n_epochs": 300,
    "lr": 1e-4,
    "val_interval": 5,
    "num_workers": 8,
    "num_train_timesteps": 1000,
    "spatial_dims": 3,
    "latent_channels": 4,
    "model_channels": (64, 128, 256, 512),
    "attention_levels": (False, False, True, True),
    "num_res_blocks": 2,
    "num_head_channels": 64,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def build_model(config, device):
    """Instantiate the DiffusionModelUNet and wrap in DataParallel if needed."""
    model = DiffusionModelUNet(
        spatial_dims=config["spatial_dims"],
        in_channels=config["latent_channels"] * 2,   # noisy target + condition
        out_channels=config["latent_channels"],
        channels=config["model_channels"],
        attention_levels=config["attention_levels"],
        num_res_blocks=config["num_res_blocks"],
        num_head_channels=config["num_head_channels"],
    )

    if torch.cuda.device_count() > 1:
        print(f"Wrapping model in DataParallel on {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    return model.to(device)


def build_scheduler(config):
    """Return the DDPM noise scheduler."""
    return DDPMScheduler(
        num_train_timesteps=config["num_train_timesteps"],
        beta_schedule="squaredcos_cap_v2",
        prediction_type="sample",
        clip_sample=False,
    )


def prepare_dataloaders(config):
    """Split patients 80/10/10 and return train/val DataLoaders."""
    all_patient_dirs = sorted(glob.glob(os.path.join(config["data_dir"], "Patient_*")))
    all_patients = [os.path.basename(d) for d in all_patient_dirs]

    np.random.seed(42)
    np.random.shuffle(all_patients)

    n_train = int(0.8 * len(all_patients))
    n_val = int(0.1 * len(all_patients))

    train_patients = all_patients[:n_train]
    val_patients = all_patients[n_train : n_train + n_val]

    print(f"Train: {len(train_patients)}, Val: {len(val_patients)}, "
          f"Test: {len(all_patients) - n_train - n_val}")

    train_ds = LatentPairsDataset(config["data_dir"], patient_split=train_patients)
    val_ds = LatentPairsDataset(config["data_dir"], patient_split=val_patients)

    train_loader = DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=True,
        num_workers=config["num_workers"], pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config["batch_size"], shuffle=False,
        num_workers=config["num_workers"], pin_memory=True,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(config):
    set_determinism(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}  |  GPU count: {torch.cuda.device_count()}")

    os.makedirs(config["save_dir"], exist_ok=True)

    # Data
    train_loader, val_loader = prepare_dataloaders(config)

    # Model & scheduler
    model = build_model(config, device)
    scheduler = build_scheduler(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    scaler = torch.cuda.amp.GradScaler()

    # Latent scale factor (computed from first batch)
    first_batch = next(iter(train_loader))
    _, first_fn = first_batch
    latent_std = first_fn.flatten().std().item()
    scale_factor = 1.0 / latent_std
    print(f"Latent Std: {latent_std:.4f}  |  Scale Factor: {scale_factor:.4f}")

    epoch_loss_values = []
    val_loss_values = []

    for epoch in range(config["n_epochs"]):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['n_epochs']}")

        for f1, fn in progress_bar:
            f1 = f1.to(device)
            fn = fn.to(device)

            f1_scaled = f1 * scale_factor
            fn_scaled = fn * scale_factor

            optimizer.zero_grad()

            timesteps = torch.randint(
                0, scheduler.num_train_timesteps, (fn.shape[0],), device=device
            ).long()

            noise = torch.randn_like(fn_scaled)
            noisy_fn = scheduler.add_noise(original_samples=fn_scaled, noise=noise, timesteps=timesteps)

            model_input = torch.cat([noisy_fn, f1_scaled], dim=1)

            with torch.cuda.amp.autocast():
                model_output = model(model_input, timesteps=timesteps)
                loss = F.mse_loss(model_output, fn_scaled)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

        avg_loss = epoch_loss / len(train_loader)
        epoch_loss_values.append(avg_loss)

        # Validation
        if (epoch + 1) % config["val_interval"] == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for f1, fn in val_loader:
                    f1 = f1.to(device)
                    fn = fn.to(device)
                    f1_scaled = f1 * scale_factor
                    fn_scaled = fn * scale_factor

                    timesteps = torch.randint(
                        0, scheduler.num_train_timesteps, (fn.shape[0],), device=device
                    ).long()
                    noise = torch.randn_like(fn_scaled)
                    noisy_fn = scheduler.add_noise(original_samples=fn_scaled, noise=noise, timesteps=timesteps)
                    model_input = torch.cat([noisy_fn, f1_scaled], dim=1)

                    with torch.cuda.amp.autocast():
                        model_output = model(model_input, timesteps=timesteps)
                        loss = F.mse_loss(model_output, fn_scaled)

                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            val_loss_values.append(avg_val_loss)
            print(f"Epoch {epoch + 1} — Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

            torch.save(
                model.state_dict(),
                os.path.join(config["save_dir"], f"model_epoch_{epoch + 1}.pt"),
            )

    print("\nTraining complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train Conditional Latent Diffusion Model")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_CONFIG["data_dir"])
    parser.add_argument("--save-dir", type=str, default=DEFAULT_CONFIG["save_dir"])
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["n_epochs"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["lr"])
    args = parser.parse_args()

    config = {**DEFAULT_CONFIG}
    config["data_dir"] = args.data_dir
    config["save_dir"] = args.save_dir
    config["n_epochs"] = args.epochs
    config["batch_size"] = args.batch_size
    config["lr"] = args.lr

    train(config)


if __name__ == "__main__":
    main()
