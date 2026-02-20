"""
Evaluation metrics for comparing GT and predicted probability maps.

Metrics
-------
- **Gray-level Dice** — soft overlap between two probability maps.
- **ADICE (Adaptive Dice)** — mean binary Dice across 10 thresholds.
- **GED (Generalised Energy Distance)** — distribution-level metric
  based on IoU between sample sets.

Usage:
    python metrics.py [--output-dir DIR] [--data-dir DIR]
"""

import os
import glob
import json
import time
import argparse

import numpy as np
import nibabel as nib
import SimpleITK as sitk
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STRUCTURE_LABELS = [2, 3, 4, 5]
STRUCTURE_NAMES = {
    2: "rectum",
    3: "bladder",
    4: "prostate",
    5: "femur_heads",
}

DEFAULT_DATA_DIR = "/net/tscratch/people/plgztabor/ROBUST_PLANNING/DATA"
DEFAULT_OUTPUT_DIR = "/net/tscratch/people/plgpiotreksl/csd_verification_results_fixed"

TEST_IDS = {
    "02", "03", "17", "24", "25", "27", "33",
    "42", "48", "57", "59", "68", "71", "76", "77",
}


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------
def iou(x, y):
    """Intersection over Union between two masks."""
    union = x + y
    union[union > 0] = 1
    x_bin = np.copy(x)
    x_bin[x_bin > 0] = 1
    y_bin = np.copy(y)
    y_bin[y_bin > 0] = 1
    return np.sum(x_bin * y_bin) / np.sum(union)


def gray_level_dice(gt, pred):
    """Gray-level (soft) Dice between two probability maps."""
    return 2 * np.sum(np.sqrt(gt * pred)) / (np.sum(gt) + np.sum(pred))


def adaptive_dice(gt, pred):
    """ADICE — mean binary Dice across thresholds 0.1 … 1.0."""
    thresholds = [0.1 + i * 0.1 for i in range(10)]
    scores = []
    for th in thresholds:
        gt_bin = (gt >= th).astype(np.float32)
        pred_bin = (pred >= th).astype(np.float32)
        denom = gt_bin.sum() + pred_bin.sum()
        scores.append(2 * np.sum(gt_bin * pred_bin) / denom if denom > 0 else 0.0)
    return np.mean(scores)


def compute_ged(gt_imgs, pred_imgs):
    """
    Generalised Energy Distance (GED) between two sets of binary masks.

    GED = 2 * E[d(gt,pred)] - E[d(gt,gt)] - E[d(pred,pred)]
    where d = 1 - IoU.
    """
    # E[d(gt, pred)]
    s1 = 0.0
    for g in gt_imgs:
        for p in pred_imgs:
            s1 += 1 - iou(g, p)
    s1 /= len(gt_imgs) * len(pred_imgs)

    # E[d(gt, gt)]
    s2 = 0.0
    n_gt = len(gt_imgs)
    for i in range(n_gt - 1):
        for j in range(i + 1, n_gt):
            s2 += 1 - iou(gt_imgs[i], gt_imgs[j])
    s2 /= n_gt * (n_gt - 1) / 2

    # E[d(pred, pred)]
    s3 = 0.0
    n_pred = len(pred_imgs)
    for i in range(n_pred - 1):
        for j in range(i + 1, n_pred):
            s3 += 1 - iou(pred_imgs[i], pred_imgs[j])
    s3 /= n_pred * (n_pred - 1) / 2

    return 2 * s1 - s2 - s3


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------
def evaluate(data_dir, output_dir):
    test_patient_ids = sorted(TEST_IDS)

    dices = {s: [] for s in STRUCTURE_LABELS}
    adices = {s: [] for s in STRUCTURE_LABELS}
    geds = {s: [] for s in STRUCTURE_LABELS}

    total_start = time.time()

    for sid in STRUCTURE_LABELS:
        print(f"\n{'=' * 60}\nStructure {sid} ({STRUCTURE_NAMES[sid]})\n{'=' * 60}")

        for pid in tqdm(test_patient_ids, desc=f"Structure {sid}"):
            gt_path = (
                f"{output_dir}/probability_maps/"
                f"GT_Patient_{pid}_Structure_{sid}_{STRUCTURE_NAMES[sid]}.nii.gz"
            )
            pred_path = (
                f"{output_dir}/probability_maps/"
                f"PRED_SD_Patient_{pid}_Structure_{sid}_{STRUCTURE_NAMES[sid]}.nii.gz"
            )
            if not os.path.exists(gt_path) or not os.path.exists(pred_path):
                print(f"  Missing files for patient {pid}")
                continue

            gt = nib.load(gt_path).get_fdata()
            pred = nib.load(pred_path).get_fdata()

            # Gray-level DICE
            dice_val = gray_level_dice(gt, pred)
            dices[sid].append(float(dice_val))

            # ADICE
            adice_val = adaptive_dice(gt, pred)
            adices[sid].append(float(adice_val))

            # GED — requires individual masks
            struct_dir = f"{data_dir}/STRUCTURES/Patient_{pid}"
            struct_files = sorted(
                glob.glob(f"{struct_dir}/Patient_{pid}_fraction_*.nii.gz")
            )
            gt_imgs = []
            for sf in struct_files:
                d = (
                    nib.load(sf)
                    .get_fdata()
                    .swapaxes(2, 1)
                    .swapaxes(1, 0)
                    .swapaxes(2, 1)
                )
                gt_imgs.append((d == sid).astype(np.float32))

            warped_pattern = (
                f"{output_dir}/warped_structures/"
                f"warped_Patient_{pid}_sample_*_hu_Structure_{sid}.nii.gz"
            )
            warped_files = sorted(glob.glob(warped_pattern))
            pred_imgs = [
                (sitk.GetArrayFromImage(sitk.ReadImage(wf)) >= 0.5).astype(np.float32)
                for wf in warped_files
            ]

            if len(gt_imgs) >= 2 and len(pred_imgs) >= 2:
                ged_val = compute_ged(gt_imgs, pred_imgs)
                geds[sid].append(float(ged_val))
            else:
                geds[sid].append(None)

            print(
                f"  Patient {pid}: DICE={dice_val:.4f}, ADICE={adice_val:.4f}, "
                f"GED={geds[sid][-1]}"
            )

        # Per-structure summary
        valid_geds = [g for g in geds[sid] if g is not None]
        if dices[sid]:
            ged_info = (
                f"GED={np.mean(valid_geds):.4f}+/-{np.std(valid_geds):.4f}"
                if valid_geds
                else "GED=N/A"
            )
            print(
                f"\n  SUMMARY — DICE={np.mean(dices[sid]):.4f}+/-{np.std(dices[sid]):.4f}  "
                f"ADICE={np.mean(adices[sid]):.4f}+/-{np.std(adices[sid]):.4f}  "
                f"{ged_info}"
            )

    # Save results
    results_path = f"{output_dir}/results.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "ids": list(test_patient_ids),
                "dices": {int(k): v for k, v in dices.items()},
                "adices": {int(k): v for k, v in adices.items()},
                "geds": {int(k): v for k, v in geds.items()},
            },
            f,
            indent=4,
        )

    total_time = time.time() - total_start
    print(f"\nTotal time: {total_time / 60:.1f} min")
    print(f"Results saved to: {results_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate probability-map metrics")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    evaluate(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
