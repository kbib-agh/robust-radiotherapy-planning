"""
Dataset classes for the Stable Diffusion training and generation pipeline.

Contains:
- LatentPairsDataset: Pairs (fraction_1, fraction_n) for diffusion model training
- PatientConditionDataset: Returns fraction_1 condition for sample generation
"""

import os
import glob
import torch
from torch.utils.data import Dataset


class LatentPairsDataset(Dataset):
    """Dataset that pairs fraction_1 with fraction_n for each patient.

    Used during diffusion model training to provide (condition, target) latent pairs.

    Args:
        root_dir: Path to directory containing per-patient latent encodings.
        patient_split: List of patient directory names to include, or None for all.
    """

    def __init__(self, root_dir, patient_split=None):
        self.root_dir = root_dir
        self.pairs = []

        all_patient_dirs = sorted(glob.glob(os.path.join(root_dir, "Patient_*")))

        if patient_split is not None:
            patient_dirs = [d for d in all_patient_dirs if os.path.basename(d) in patient_split]
        else:
            patient_dirs = all_patient_dirs

        for p_dir in patient_dirs:
            p_id = os.path.basename(p_dir)
            f1_files = glob.glob(os.path.join(p_dir, f"{p_id}_fraction_1_.pt"))
            if not f1_files:
                f1_files = glob.glob(os.path.join(p_dir, f"{p_id}_fraction_1.pt"))

            if not f1_files:
                continue

            f1_path = f1_files[0]

            all_fractions = sorted(glob.glob(os.path.join(p_dir, f"{p_id}_fraction_*.pt")))
            for f_path in all_fractions:
                if f_path == f1_path:
                    continue
                self.pairs.append((f1_path, f_path))

        print(f"Found {len(self.pairs)} pairs across {len(patient_dirs)} patients.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        f1_path, fn_path = self.pairs[idx]

        f1_data = torch.load(f1_path, map_location="cpu", weights_only=False)
        fn_data = torch.load(fn_path, map_location="cpu", weights_only=False)

        f1 = f1_data["latent"] if isinstance(f1_data, dict) and "latent" in f1_data else f1_data
        fn = fn_data["latent"] if isinstance(fn_data, dict) and "latent" in fn_data else fn_data

        if isinstance(f1, dict):
            f1 = f1.get("z", list(f1.values())[0])
        if isinstance(fn, dict):
            fn = fn.get("z", list(fn.values())[0])

        # Ensure [C, D, H, W]
        if f1.ndim == 5 and f1.shape[0] == 1:
            f1 = f1.squeeze(0)
        if fn.ndim == 5 and fn.shape[0] == 1:
            fn = fn.squeeze(0)
        if f1.ndim == 3:
            f1 = f1.unsqueeze(0)
        if fn.ndim == 3:
            fn = fn.unsqueeze(0)

        return f1.float(), fn.float()


class PatientConditionDataset(Dataset):
    """Dataset that returns fraction_1 (condition) for each patient.

    Used during sample generation to iterate over test patients.

    Args:
        root_dir: Path to directory containing per-patient latent encodings.
        patient_split: Set of patient IDs to include, or None for all.
    """

    def __init__(self, root_dir, patient_split=None):
        self.root_dir = root_dir
        self.samples = []

        patient_dirs = sorted(glob.glob(os.path.join(root_dir, "Patient_*")))

        if patient_split is not None:
            filtered_dirs = []
            for p_dir in patient_dirs:
                p_id = os.path.basename(p_dir).split("_")[1]
                if p_id in patient_split:
                    filtered_dirs.append(p_dir)
            patient_dirs = filtered_dirs

        for p_dir in patient_dirs:
            fractions = sorted(glob.glob(os.path.join(p_dir, "*.pt")))
            f1_path = None
            for f in fractions:
                if "fraction_1_" in os.path.basename(f):
                    f1_path = f
                    break
            if f1_path is not None:
                self.samples.append(f1_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        f1_path = self.samples[idx]
        f1_data = torch.load(f1_path, map_location="cpu")

        if isinstance(f1_data, dict):
            f1 = f1_data.get("latent", f1_data.get("z", list(f1_data.values())[0]))
        else:
            f1 = f1_data

        f1 = f1.float()
        if len(f1.shape) == 5:
            f1 = f1.squeeze(0)

        return f1, f1_path
