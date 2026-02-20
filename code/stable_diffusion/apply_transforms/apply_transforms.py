"""
Apply displacement-field transforms to compute predicted probability maps
and compare them with ground-truth probability maps.

Workflow
-------
1.  Compute **GT probability maps** by averaging binary masks across all
    real fractions for each patient.
2.  Compute **predicted probability maps** by averaging the warped
    fraction-1 structures (transformed via displacement fields produced
    during registration).

Usage:
    python apply_transforms.py [--data-dir DIR] [--output-dir DIR]
"""

import os
import glob
import json
import argparse

import numpy as np
import nibabel as nib
import SimpleITK as sitk
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SPACING = (1.171875, 1.171875, 3.0)
ORIGIN = (0.0, 0.0, 0.0)
DIRECTION = (-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0)
AFF = np.eye(4)

STRUCTURE_LABELS = [2, 3, 4, 5]
STRUCTURE_NAMES = {
    2: "rectum",
    3: "bladder",
    4: "prostate",
    5: "femur_heads",
}

NUM_SAMPLES = 10

DEFAULT_DATA_DIR = "/net/tscratch/people/plgztabor/ROBUST_PLANNING/DATA"
DEFAULT_OUTPUT_DIR = "/net/tscratch/people/plgpiotreksl/csd_verification_results_fixed"

TEST_IDS = {
    "02", "03", "17", "24", "25", "27", "33",
    "42", "48", "57", "59", "68", "71", "76", "77",
}


# ---------------------------------------------------------------------------
# Probability-map computation
# ---------------------------------------------------------------------------
def compute_probability_map(masks_list):
    """Average a list of binary (or soft) masks to obtain a probability map."""
    return np.mean(np.stack(masks_list, axis=0), axis=0)


def compute_gt_probability_map(patient_id, structure_label, data_dir):
    """
    Build the ground-truth probability map for *structure_label* by
    averaging the binary mask across all real fractions.
    """
    structure_dir = f"{data_dir}/STRUCTURES/Patient_{patient_id}"
    structure_files = sorted(
        glob.glob(f"{structure_dir}/Patient_{patient_id}_fraction_*.nii.gz")
    )
    if not structure_files:
        print(f"No structure files for patient {patient_id}")
        return None

    masks = []
    for sf in structure_files:
        data = (
            nib.load(sf)
            .get_fdata()
            .swapaxes(2, 1)
            .swapaxes(1, 0)
            .swapaxes(2, 1)
        )
        masks.append((data == structure_label).astype(np.float32))

    return compute_probability_map(masks)


def compute_predicted_probability_maps(patient_id, output_dir):
    """
    Load all warped structures for *patient_id* and average them into
    predicted probability maps per structure.
    """
    predicted = {}
    for label in STRUCTURE_LABELS:
        pattern = (
            f"{output_dir}/warped_structures/"
            f"warped_Patient_{patient_id}_sample_*_hu_Structure_{label}.nii.gz"
        )
        warped_files = sorted(glob.glob(pattern))
        if not warped_files:
            continue
        arrays = [sitk.GetArrayFromImage(sitk.ReadImage(wf)) for wf in warped_files]
        predicted[label] = compute_probability_map(arrays)
    return predicted


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run(data_dir, output_dir):
    os.makedirs(f"{output_dir}/probability_maps", exist_ok=True)

    # Determine test patients with available generated samples
    generated_dir = os.path.join(
        os.path.dirname(output_dir.rstrip("/")),
        "generated_samples",
        "images_hu",
    )
    generated_files = glob.glob(
        f"{output_dir}/warped_structures/warped_Patient_*_sample_*_hu_Structure_2.nii.gz"
    )
    available_ids = {os.path.basename(f).split("_")[2] for f in generated_files}
    test_patient_ids = sorted(TEST_IDS & available_ids)
    print(f"Test patients ({len(test_patient_ids)}): {test_patient_ids}")

    # --- Step 1: GT probability maps ---
    print("\n--- Computing GT probability maps ---")
    for pid in tqdm(test_patient_ids, desc="GT maps"):
        for label in STRUCTURE_LABELS:
            save_path = (
                f"{output_dir}/probability_maps/"
                f"GT_Patient_{pid}_Structure_{label}_{STRUCTURE_NAMES[label]}.nii.gz"
            )
            if os.path.exists(save_path):
                continue
            gt = compute_gt_probability_map(pid, label, data_dir)
            if gt is not None:
                nib.save(nib.Nifti1Image(gt, affine=AFF), save_path)

    # --- Step 2: Predicted probability maps ---
    print("\n--- Computing predicted probability maps ---")
    for pid in tqdm(test_patient_ids, desc="Pred maps"):
        pred_maps = compute_predicted_probability_maps(pid, output_dir)
        for label, prob_map in pred_maps.items():
            save_path = (
                f"{output_dir}/probability_maps/"
                f"PRED_SD_Patient_{pid}_Structure_{label}_{STRUCTURE_NAMES[label]}.nii.gz"
            )
            if os.path.exists(save_path):
                continue
            nib.save(nib.Nifti1Image(prob_map, affine=AFF), save_path)

    print(f"\nProbability maps saved to: {output_dir}/probability_maps/")


def main():
    parser = argparse.ArgumentParser(description="Compute probability maps")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    run(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
