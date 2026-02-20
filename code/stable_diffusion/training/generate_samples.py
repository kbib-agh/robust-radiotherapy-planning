"""
Generate latent samples using a trained Conditional Diffusion Model,
then decode them into 3D CT images using MAISI VAE.

The script is split into two phases to manage GPU memory:
  1. **generate** — produce latent representations and save to disk.
  2. **decode**  — load saved latents, decode with VAE, rescale, and
                   restore HU values.

Usage:
    # Phase 1 – generation
    python generate_samples.py generate [--data-dir DIR] [--output-dir DIR] ...

    # Phase 2 – decoding (restart kernel / new process to free GPU)
    python generate_samples.py decode [--output-dir DIR] [--vae-checkpoint PATH] ...
"""

import os
import sys
import glob
import json
import argparse

import torch
import numpy as np
import nibabel as nib
from tqdm import tqdm
from torch.utils.data import DataLoader

from monai.networks.nets import DiffusionModelUNet
from monai.utils import set_determinism
from diffusers import DDPMScheduler

from dataset import PatientConditionDataset

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    "data_dir": "/net/tscratch/people/plgpiotreksl/maisi_vae_latent",
    "output_dir": "/net/tscratch/people/plgpiotreksl/generated_samples",
    "diffusion_checkpoint": "/net/tscratch/people/plgpiotreksl/models/conditional_diffusion_unetv2/model_best.pt",
    "vae_checkpoint": "/net/people/plgrid/plgpiotreksl/jupyter/models/autoencoder_epoch273.pt",
    "encoding_summary": "/net/tscratch/people/plgpiotreksl/maisi_vae_latent/encoding_summary.json",
    "num_samples_per_patient": 10,
    "num_inference_steps": 1000,
    "latent_scale_factor": 2.2562,
    # Model architecture
    "spatial_dims": 3,
    "latent_channels": 4,
    "model_channels": (64, 128, 256, 512),
    "attention_levels": (False, False, True, True),
    "num_res_blocks": 2,
    "num_head_channels": 64,
    # HU restoration
    "hu_min": -1000,
    "hu_max": 1000,
}

# Test patient IDs
TEST_IDS = {
    "02", "03", "17", "24", "25", "27", "33",
    "42", "48", "57", "59", "68", "71", "76", "77",
}


# ---------------------------------------------------------------------------
# Phase 1 — Generation
# ---------------------------------------------------------------------------
def generate_sample(f1, diffusion_model, scheduler, device, scale_factor):
    """Generate a single sample via reverse diffusion (DDPM, 1000 steps)."""
    f1 = f1.to(device)
    f1_scaled = f1 * scale_factor

    latent_noisy = torch.randn(f1.shape, device=device)

    with torch.no_grad():
        for t in tqdm(scheduler.timesteps, desc="Sampling", leave=False):
            model_input = torch.cat([latent_noisy, f1_scaled], dim=1)
            model_output = diffusion_model(
                model_input, timesteps=torch.tensor([t], device=device)
            )
            latent_noisy = scheduler.step(model_output, t, latent_noisy).prev_sample

    generated_latent = latent_noisy / scale_factor
    return generated_latent


def run_generation(config):
    """Phase 1: generate latents for every test patient."""
    set_determinism(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(os.path.join(config["output_dir"], "latents"), exist_ok=True)

    # Dataset
    test_ds = PatientConditionDataset(config["data_dir"], patient_split=TEST_IDS)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    print(f"Test patients found: {len(test_ds)}")

    # Model
    diffusion_model = DiffusionModelUNet(
        spatial_dims=config["spatial_dims"],
        in_channels=config["latent_channels"] * 2,
        out_channels=config["latent_channels"],
        channels=config["model_channels"],
        attention_levels=config["attention_levels"],
        num_res_blocks=config["num_res_blocks"],
        num_head_channels=config["num_head_channels"],
    ).to(device)

    # Load checkpoint (handle DataParallel keys)
    state_dict = torch.load(config["diffusion_checkpoint"], map_location="cpu")
    new_state_dict = {
        k.replace("module.", ""): v for k, v in state_dict.items()
    }
    diffusion_model.load_state_dict(new_state_dict)
    diffusion_model.eval()
    print("Diffusion model loaded.")

    scheduler = DDPMScheduler(
        num_train_timesteps=config["num_inference_steps"],
        beta_schedule="squaredcos_cap_v2",
        prediction_type="sample",
        clip_sample=False,
    )

    scale_factor = config["latent_scale_factor"]

    for _i, batch in enumerate(tqdm(test_loader, desc="Test patients")):
        f1, f1_path = batch
        patient_id = os.path.basename(os.path.dirname(f1_path[0])).split("_")[1]

        for sample_idx in range(config["num_samples_per_patient"]):
            base_filename = f"Patient_{patient_id}_sample_{sample_idx + 1}"
            latent_save_path = os.path.join(
                config["output_dir"], "latents", f"{base_filename}.pt"
            )
            if os.path.exists(latent_save_path):
                print(f"Skipping {base_filename}, already exists.")
                continue

            print(f"Generating {base_filename}...")
            generated_latent = generate_sample(
                f1, diffusion_model, scheduler, device, scale_factor
            )
            torch.save(generated_latent.cpu(), latent_save_path)

    print("Generation complete.")


# ---------------------------------------------------------------------------
# Phase 2 — Decoding, Rescaling, HU Restoration
# ---------------------------------------------------------------------------
def decode_latent(latent, vae, device):
    """Decode a single latent tensor to an image using the MAISI VAE."""
    latent = latent.to(device)
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True):
            decoded = vae.decode(latent)
        if isinstance(decoded, tuple):
            decoded = decoded[0]
    return decoded


def get_original_shape(patient_id, summary_data):
    """Look up the original CT shape for *patient_id* from the encoding summary."""
    patients_list = summary_data.get("patient_stats", summary_data)
    if isinstance(patients_list, dict):
        patients_list = patients_list.get("patient_stats", [])
    for entry in patients_list:
        if entry.get("patient_id") == patient_id:
            for frac in entry.get("fractions", []):
                if "fraction_1_" in frac.get("name", ""):
                    return frac["original_shape"]
    return None


def restore_hu_values(normalized_image, hu_min=-1000, hu_max=1000):
    """Reverse the ScaleIntensityRanged [0,1] -> [hu_min, hu_max]."""
    return normalized_image * (hu_max - hu_min) + hu_min


def run_decoding(config):
    """Phase 2: decode latents, rescale, restore HU values."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load VAE ---
    sys.path.append("/net/people/plgrid/plgpiotreksl/maisi/scripts")
    from argparse import Namespace
    from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi  # noqa: F401
    from utils import define_instance

    vae_args = Namespace()
    vae_args.autoencoder_def = {
        "_target_": "monai.apps.generation.maisi.networks.autoencoderkl_maisi.AutoencoderKlMaisi",
        "spatial_dims": 3,
        "in_channels": 1,
        "out_channels": 1,
        "latent_channels": 4,
        "num_channels": [64, 128, 256],
        "num_res_blocks": [2, 2, 2],
        "norm_num_groups": 32,
        "norm_eps": 1e-06,
        "attention_levels": [False, False, False],
        "with_encoder_nonlocal_attn": False,
        "with_decoder_nonlocal_attn": False,
        "use_checkpointing": False,
        "use_convtranspose": False,
        "save_mem": True,
        "norm_float16": False,
        "num_splits": 32,
        "dim_split": 0,
    }
    vae = define_instance(vae_args, "autoencoder_def").to(device)
    vae.load_state_dict(torch.load(config["vae_checkpoint"], map_location=device))
    vae.eval()
    print("VAE loaded.")

    # --- Decode ---
    latents_dir = os.path.join(config["output_dir"], "latents")
    images_dir = os.path.join(config["output_dir"], "images")
    os.makedirs(images_dir, exist_ok=True)

    latent_files = sorted(glob.glob(os.path.join(latents_dir, "*.pt")))
    print(f"Found {len(latent_files)} latent files.")

    decoded_images = []
    for lp in tqdm(latent_files, desc="Decoding"):
        base_name = os.path.basename(lp).replace(".pt", "")
        out_path = os.path.join(images_dir, f"{base_name}.nii.gz")
        if os.path.exists(out_path):
            decoded_images.append((base_name, out_path))
            continue

        latent = torch.load(lp, map_location="cpu")
        if len(latent.shape) == 4:
            latent = latent.unsqueeze(0)

        decoded = decode_latent(latent, vae, device)
        img_data = decoded.squeeze().cpu().float().numpy()
        nib.save(nib.Nifti1Image(img_data, affine=np.eye(4)), out_path)
        decoded_images.append((base_name, out_path))

    print("Decoding complete.")

    # --- Rescale to original resolution ---
    encoding_summary_path = config["encoding_summary"]
    with open(encoding_summary_path, "r") as f:
        encoding_summary = json.load(f)

    from monai.transforms import Resize

    rescaled_dir = os.path.join(config["output_dir"], "images_rescaled")
    os.makedirs(rescaled_dir, exist_ok=True)

    rescaled_images = []
    for base_name, img_path in tqdm(decoded_images, desc="Rescaling"):
        patient_id = base_name.split("_")[1]
        org_shape = get_original_shape(patient_id, encoding_summary)
        if org_shape is None:
            print(f"Warning: no original shape for Patient {patient_id}")
            continue

        nifti_img = nib.load(img_path)
        img_tensor = torch.tensor(nifti_img.get_fdata()).unsqueeze(0)
        resizer = Resize(spatial_size=org_shape, mode="trilinear")
        rescaled = resizer(img_tensor).squeeze(0).numpy()

        save_path = os.path.join(rescaled_dir, f"{base_name}_rescaled.nii.gz")
        nib.save(nib.Nifti1Image(rescaled, affine=nifti_img.affine), save_path)
        rescaled_images.append((base_name, save_path))

    print("Rescaling complete.")

    # --- Restore HU ---
    hu_dir = os.path.join(config["output_dir"], "images_hu")
    os.makedirs(hu_dir, exist_ok=True)

    hu_min, hu_max = config["hu_min"], config["hu_max"]
    for base_name, img_path in tqdm(rescaled_images, desc="Restoring HU"):
        nifti_img = nib.load(img_path)
        hu_data = restore_hu_values(nifti_img.get_fdata(), hu_min, hu_max)
        out_path = os.path.join(hu_dir, f"{base_name}_hu.nii.gz")
        nib.save(nib.Nifti1Image(hu_data.astype(np.float32), affine=nifti_img.affine), out_path)

    print(f"HU restoration complete. Images saved to: {hu_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate & decode latent diffusion samples")
    subparsers = parser.add_subparsers(dest="phase", required=True)

    # Phase 1
    gen_parser = subparsers.add_parser("generate", help="Generate latent samples")
    gen_parser.add_argument("--data-dir", type=str, default=DEFAULT_CONFIG["data_dir"])
    gen_parser.add_argument("--output-dir", type=str, default=DEFAULT_CONFIG["output_dir"])
    gen_parser.add_argument("--diffusion-checkpoint", type=str, default=DEFAULT_CONFIG["diffusion_checkpoint"])
    gen_parser.add_argument("--num-samples", type=int, default=DEFAULT_CONFIG["num_samples_per_patient"])

    # Phase 2
    dec_parser = subparsers.add_parser("decode", help="Decode latents to images")
    dec_parser.add_argument("--output-dir", type=str, default=DEFAULT_CONFIG["output_dir"])
    dec_parser.add_argument("--vae-checkpoint", type=str, default=DEFAULT_CONFIG["vae_checkpoint"])
    dec_parser.add_argument("--encoding-summary", type=str, default=DEFAULT_CONFIG["encoding_summary"])

    args = parser.parse_args()

    config = {**DEFAULT_CONFIG}

    if args.phase == "generate":
        config["data_dir"] = args.data_dir
        config["output_dir"] = args.output_dir
        config["diffusion_checkpoint"] = args.diffusion_checkpoint
        config["num_samples_per_patient"] = args.num_samples
        run_generation(config)
    elif args.phase == "decode":
        config["output_dir"] = args.output_dir
        config["vae_checkpoint"] = args.vae_checkpoint
        config["encoding_summary"] = args.encoding_summary
        run_decoding(config)


if __name__ == "__main__":
    main()
