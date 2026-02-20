# Stable Diffusion Approach

This module implements anatomical variant generation using a **Latent Diffusion Model** conditioned on the first fraction CT scan. Generated variants are used to estimate the probability distribution of 3D dose maps for robust radiotherapy planning.

---

## Pipeline Overview

The pipeline consists of five sequential stages:

```
encode_maisi_vae.py → conditional_diffusion_unet.ipynb → generate_samples.ipynb → CSDVerification.ipynb → DoseUncertainty.ipynb
```

### 1. VAE Encoding (`encode_maisi_vae.py`)

Compresses 3D CT volumes into compact latent representations using the **MAISI VAE** (AutoencoderKlMaisi from MONAI).

- **Input**: NIfTI CT scans per patient/fraction
- **Output**: Latent tensors (`.pt` files) with shape `[1, 4, D, H, W]`
- **Transform**: `ScaleIntensityRanged(a_min=-1000, a_max=1000, b_min=0, b_max=1)` normalizes HU values before encoding
- Saves an `encoding_summary.json` with original shapes for later rescaling

### 2. Diffusion Model Training (`conditional_diffusion_unet.ipynb`)

Trains a **Conditional 3D Diffusion UNet** (`monai.networks.nets.DiffusionModelUNet`) to predict future fraction latents conditioned on the first fraction latent.

- **Input**: Pairs of `(fraction_1_latent, fraction_n_latent)` per patient
- **Conditioning**: Channel concatenation — 4 channels (noisy target) + 4 channels (condition)
- **Scheduler**: DDPM with cosine noise schedule, predicts clean sample (x₀)
- **Hardware**: Multi-GPU training via `DataParallel` (optimized for 4× A100 40GB)
- **Output**: Model checkpoint (`model_best.pt`)

### 3. Sample Generation (`generate_samples.ipynb`)

Generates 10 latent samples per test patient using the trained diffusion model, then decodes them into 3D CT images.

- **Part 1 — Generation**: Runs inference with 1000 DDPM steps per sample, saves latent tensors to disk
- **Part 2 — Decoding** (after kernel restart): Loads the MAISI VAE, decodes latents to images, rescales to original resolution, and restores HU values
- **Output**: NIfTI images in `images/`, `images_rescaled/`, and `images_hu/` subdirectories

### 4. Verification (`CSDVerification.ipynb`)

Validates generated samples by registering them to reference fraction CTs and computing structure overlap metrics.

- Registers each generated sample to the planning CT using **SimpleITK** diffeomorphic demons
- Warps OAR structure masks using the resulting deformation fields
- Computes **probability maps** for each structure across all samples
- Evaluates metrics: **Dice**, **Average Dice (ADICE)**, and **Generalized Energy Distance (GED)**
- **Structures evaluated**: `rectum`, `bladder`, `prostate`, `femur_heads`

### 5. Dose Uncertainty Analysis (`DoseUncertainty.ipynb`)

Estimates dose distribution uncertainty from generated anatomical variants.

- Loads pre-computed dose distributions for each generated sample
- Computes per-voxel **mean dose** and **dose standard deviation** maps
- Calculates **DVH (Dose-Volume Histogram)** statistics per structure
- Compares uncertainty metrics against the DDF-based approach
- Visualizes spatial uncertainty maps across axial, coronal, and sagittal planes
- **Structures analyzed**: `rectum`, `bladder`, `prostate`, `femur_heads`

---

## Structure Mapping

The structure label indices map to the following organs at risk:

| Label | Structure    |
|-------|-------------|
| 2     | rectum       |
| 3     | bladder      |
| 4     | prostate     |
| 5     | femur_heads  |

---

## Output Files

| File | Description |
|------|-------------|
| `results.json` | Quantitative metrics (Dice, ADICE, GED) per patient |
| `dose_stats_stable_diffusion.txt` | DVH statistics per structure across all patients |

---

## Requirements

- Python 3.8+
- PyTorch
- MONAI (with `DiffusionModelUNet` and MAISI VAE support)
- SimpleITK
- nibabel
- diffusers (`DDPMScheduler`)
- matplotlib, numpy, scipy
