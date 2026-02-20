# Stable Diffusion Approach

This module implements anatomical variant generation using a **Latent Diffusion Model** conditioned on the first fraction CT scan. Generated variants are used to estimate the probability distribution of 3D dose maps for robust radiotherapy planning.

---

## Folder Structure

```
stable_diffusion/
├── training/                   # Model training and sample generation
│   ├── encode_vae.py           # VAE encoding of CT volumes to latent space
│   ├── dataset.py              # Dataset classes (LatentPairsDataset, PatientConditionDataset)
│   ├── train_diffusion.py      # Conditional diffusion model training
│   └── generate_samples.py     # Latent generation, VAE decoding, rescaling & HU restoration
│
├── registration/               # Image registration
│   └── registration.py         # Diffeomorphic Demons registration (SimpleITK)
│
├── apply_transforms/           # Structure warping, probability maps & evaluation
│   ├── apply_transforms.py     # Compute GT and predicted probability maps
│   ├── metrics.py              # Dice, ADICE, GED evaluation metrics
│   ├── dose_uncertainty.py     # Dose uncertainty analysis + DDF comparison
│   ├── results.json            # Quantitative metrics (Dice, ADICE, GED) per patient
│   └── dose_stats_stable_diffusion.txt  # Dose statistics in DDF-compatible format
│
├── _notebooks/                 # Original interactive notebooks
│   ├── conditional_diffusion_unet.ipynb
│   ├── generate_samples.ipynb
│   ├── CSDVerification.ipynb
│   └── DoseUncertainty.ipynb
│
└── README.md
```

---

## Pipeline Overview

The pipeline consists of five sequential stages:

```
training/encode_vae.py → training/train_diffusion.py → training/generate_samples.py → registration/registration.py + apply_transforms/ → apply_transforms/dose_uncertainty.py
```

### 1. VAE Encoding (`training/encode_vae.py`)

Compresses 3D CT volumes into compact latent representations using the **MAISI VAE** (AutoencoderKlMaisi from MONAI).

- **Input**: NIfTI CT scans per patient/fraction
- **Output**: Latent tensors (`.pt` files) with shape `[1, 4, D, H, W]`
- **Transform**: `ScaleIntensityRanged(a_min=-1000, a_max=1000, b_min=0, b_max=1)` normalizes HU values before encoding
- Saves an `encoding_summary.json` with original shapes for later rescaling

### 2. Diffusion Model Training (`training/train_diffusion.py`)

Trains a **Conditional 3D Diffusion UNet** (`monai.networks.nets.DiffusionModelUNet`) to predict future fraction latents conditioned on the first fraction latent.

- **Input**: Pairs of `(fraction_1_latent, fraction_n_latent)` per patient (via `dataset.LatentPairsDataset`)
- **Conditioning**: Channel concatenation — 4 channels (noisy target) + 4 channels (condition)
- **Scheduler**: DDPM with cosine noise schedule, predicts clean sample (x₀)
- **Hardware**: Multi-GPU training via `DataParallel` (optimized for 4× A100 40GB)
- **Output**: Model checkpoint (`model_best.pt`)

### 3. Sample Generation (`training/generate_samples.py`)

Generates 10 latent samples per test patient using the trained diffusion model, then decodes them into 3D CT images.

- **Phase 1 — `generate`**: Runs inference with 1000 DDPM steps per sample, saves latent tensors to disk
- **Phase 2 — `decode`** (separate process to free GPU): Loads the MAISI VAE, decodes latents to images, rescales to original resolution, and restores HU values
- **Output**: NIfTI images in `images/`, `images_rescaled/`, and `images_hu/` subdirectories

### 4. Registration & Structure Warping (`registration/` + `apply_transforms/`)

- **`registration/registration.py`** — Registers each generated sample to the planning CT using **SimpleITK** diffeomorphic demons, saves displacement fields
- **`apply_transforms/apply_transforms.py`** — Warps fraction-1 OAR masks through displacement fields, computes GT and predicted **probability maps**
- **`apply_transforms/metrics.py`** — Evaluates **Gray-level Dice**, **ADICE**, and **GED** between GT and predicted probability maps
- **Structures evaluated**: `rectum`, `bladder`, `prostate`, `femur_heads`

### 5. Dose Uncertainty Analysis (`apply_transforms/dose_uncertainty.py`)

Estimates dose distribution uncertainty from generated anatomical variants.

- Recalculates dose based on HU differences: `D_gen = D_plan × (1 + HU_gen/1000) / (1 + HU_plan/1000)`
- Applies displacement field transforms to recalculated doses
- Computes per-voxel **mean dose** and **dose standard deviation** maps
- Calculates per-structure statistics and compares against the DDF-based approach
- Outputs `dose_stats_stable_diffusion.txt` in DDF-compatible format
- **Structures analyzed**: `rectum`, `bladder`, `prostate`, `femur_heads`

---

## Structure Mapping

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
| `apply_transforms/results.json` | Quantitative metrics (Dice, ADICE, GED) per patient |
| `apply_transforms/dose_stats_stable_diffusion.txt` | Dose statistics in DDF-compatible format |

---

## Requirements

- Python 3.8+
- PyTorch
- MONAI (with `DiffusionModelUNet` and MAISI VAE support)
- SimpleITK
- nibabel
- diffusers (`DDPMScheduler`)
- matplotlib, numpy, scipy
