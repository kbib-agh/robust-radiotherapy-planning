"""
Dose uncertainty analysis based on generated anatomical variants.

For each test patient the script:
1.  Loads the planned dose and CT (fraction 1).
2.  For every generated sample, recalculates the dose based on HU
    differences and applies the previously computed displacement field.
3.  Aggregates the transformed doses to produce mean / std uncertainty maps
    and per-structure statistics.
4.  (Optionally) compares results with the DDF baseline stored in
    ``dose_stats.txt``.

Usage:
    python dose_uncertainty.py [--data-dir DIR] [--output-dir DIR] ...
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

# Per-patient flag: 1 = flip Z axis of the dose volume before processing.
DOSE_FLIP_FLAGS = {
    "03": 1,
    "17": 0,
    "25": 1,
    "27": 1,
    "33": 0,
    "42": 0,
    "59": 1,
    "68": 1,
    "71": 1,
    "76": 1,
    "77": 0,
}

NUM_SAMPLES = 10

DEFAULT_DATA_DIR = "/net/tscratch/people/plgztabor/ROBUST_PLANNING/DATA"
DEFAULT_GENERATED_DIR = "/net/tscratch/people/plgpiotreksl/generated_samples/images_hu"
DEFAULT_CSD_RESULTS_DIR = "/net/tscratch/people/plgpiotreksl/csd_verification_results_fixed"
DEFAULT_OUTPUT_DIR = "/net/tscratch/people/plgpiotreksl/dose_uncertainty_results_v2"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def load_nifti_as_array(filepath, swap_axes=True):
    """Load a NIfTI file and return a numpy array."""
    data = nib.load(filepath).get_fdata()
    if swap_axes:
        data = data.swapaxes(2, 1).swapaxes(1, 0).swapaxes(2, 1)
    return data


def load_nifti_as_sitk(filepath, swap_axes=True):
    """Load a NIfTI file and return a SimpleITK image."""
    data = nib.load(filepath).get_fdata()
    if swap_axes:
        data = data.swapaxes(2, 1).swapaxes(1, 0).swapaxes(2, 1)
    sitk_img = sitk.GetImageFromArray(data.astype(np.float32))
    sitk_img.SetOrigin(ORIGIN)
    sitk_img.SetSpacing(SPACING)
    sitk_img.SetDirection(DIRECTION)
    return sitk_img


def extract_structure_mask(structure_data, label):
    """Return a binary mask for *label*."""
    return (structure_data == label).astype(np.float32)


# ---------------------------------------------------------------------------
# Dose recalculation & transformation
# ---------------------------------------------------------------------------
def recalculate_dose(
    dose_planned,
    hu_planned,
    hu_generated,
    dose_threshold=0.01,
    hu_min_threshold=-500,
    ratio_limits=(0.8, 1.2),
):
    """
    Approximate dose in the generated anatomy by scaling with the HU ratio.

    D_gen = D_plan * (1 + HU_gen / 1000) / (1 + HU_plan / 1000)
    """
    dose_recalculated = dose_planned.copy()
    mask = (dose_planned > dose_threshold) & (hu_planned > hu_min_threshold)

    if not np.any(mask):
        return dose_recalculated.astype(np.float32)

    ratio = (1.0 + hu_generated[mask] / 1000.0) / (1.0 + hu_planned[mask] / 1000.0)
    ratio = np.clip(ratio, *ratio_limits)
    dose_recalculated[mask] = dose_planned[mask] * ratio
    return dose_recalculated.astype(np.float32)


def apply_transform_to_dose(dose_array, displacement_field, reference_image):
    """Resample *dose_array* using *displacement_field*."""
    dose_sitk = sitk.GetImageFromArray(dose_array.astype(np.float32))
    dose_sitk.SetOrigin(ORIGIN)
    dose_sitk.SetSpacing(SPACING)
    dose_sitk.SetDirection(DIRECTION)

    transform = sitk.DisplacementFieldTransform(displacement_field)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_image)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(transform)

    return sitk.GetArrayFromImage(resampler.Execute(dose_sitk))


# ---------------------------------------------------------------------------
# Per-patient processing
# ---------------------------------------------------------------------------
def process_patient_dose_uncertainty(
    patient_id,
    data_dir,
    generated_dir,
    csd_results_dir,
    output_dir,
    force_recalculate=False,
):
    """Process a single patient and return per-structure dose statistics."""
    print(f"\n{'=' * 60}\nProcessing patient: {patient_id}\n{'=' * 60}")

    ct_path = f"{data_dir}/CT/Patient_{patient_id}/Patient_{patient_id}_fraction_1_.nii.gz"
    dose_files = glob.glob(
        f"{data_dir}/DOSES/Patient_{patient_id}/Patient_{patient_id}_fraction_1_*.nii.gz"
    )
    struct_path = f"{data_dir}/STRUCTURES/Patient_{patient_id}/Patient_{patient_id}_fraction_1_.nii.gz"

    if not dose_files:
        print(f"No dose files for patient {patient_id}")
        return None

    hu_planned = load_nifti_as_array(ct_path)

    dose_raw = nib.load(dose_files[0]).get_fdata()
    flip = DOSE_FLIP_FLAGS.get(patient_id, 0)
    if flip:
        dose_raw = dose_raw[:, :, ::-1]
    dose_planned = dose_raw.swapaxes(2, 1).swapaxes(1, 0).swapaxes(2, 1)

    structure_data = load_nifti_as_array(struct_path)
    ct_planned_sitk = load_nifti_as_sitk(ct_path)

    sample_files = sorted(
        glob.glob(f"{generated_dir}/Patient_{patient_id}_sample_*_hu.nii.gz")
    )
    print(f"Found {len(sample_files)} generated samples")

    if not sample_files:
        return None

    os.makedirs(f"{output_dir}/recalculated_doses", exist_ok=True)
    os.makedirs(f"{output_dir}/uncertainty_maps", exist_ok=True)

    transformed_doses = []

    for sample_path in tqdm(sample_files, desc="Processing samples"):
        sample_name = os.path.basename(sample_path).replace(".nii.gz", "")
        cached_path = f"{output_dir}/recalculated_doses/transformed_dose_{sample_name}.nii.gz"

        if os.path.exists(cached_path) and not force_recalculate:
            transformed_doses.append(load_nifti_as_array(cached_path, swap_axes=False))
            continue

        transform_path = f"{csd_results_dir}/transforms/transform_{sample_name}.nii.gz"
        if not os.path.exists(transform_path):
            print(f"  Missing transform for {sample_name}")
            continue

        hu_generated = load_nifti_as_array(sample_path)
        disp_field = sitk.ReadImage(transform_path)
        disp_field.SetOrigin(ORIGIN)
        disp_field.SetSpacing(SPACING)
        disp_field.SetDirection(DIRECTION)

        dose_recalc = recalculate_dose(dose_planned, hu_planned, hu_generated)
        dose_transformed = apply_transform_to_dose(dose_recalc, disp_field, ct_planned_sitk)

        nib.save(nib.Nifti1Image(dose_transformed, affine=AFF), cached_path)
        transformed_doses.append(dose_transformed)

    if not transformed_doses:
        return None

    # Uncertainty maps
    stack = np.stack(transformed_doses, axis=0)
    dose_mean = np.mean(stack, axis=0)
    dose_std = np.std(stack, axis=0)

    nib.save(
        nib.Nifti1Image(dose_mean, affine=AFF),
        f"{output_dir}/uncertainty_maps/dose_mean_Patient_{patient_id}.nii.gz",
    )
    nib.save(
        nib.Nifti1Image(dose_std, affine=AFF),
        f"{output_dir}/uncertainty_maps/dose_std_Patient_{patient_id}.nii.gz",
    )

    # Per-structure statistics
    results = {}
    for label in STRUCTURE_LABELS:
        mask = extract_structure_mask(structure_data, label)
        if mask.sum() == 0:
            continue

        mask_bool = mask.astype(bool)
        mean_per_sample = [np.mean(d[mask_bool]) for d in transformed_doses if d.shape == mask.shape]

        if not mean_per_sample:
            continue

        results[label] = {
            "structure_name": STRUCTURE_NAMES[label],
            "mean_dose": float(np.mean(mean_per_sample)),
            "std_dose": float(np.std(mean_per_sample)),
            "mean_uncertainty": float(np.mean(dose_std[mask_bool])),
            "n_voxels": int(mask.sum()),
            "n_samples": len(mean_per_sample),
            "dose_range": [float(np.min(mean_per_sample)), float(np.max(mean_per_sample))],
        }
        print(
            f"  {STRUCTURE_NAMES[label]:20s}: Mean={results[label]['mean_dose']:.2f} Gy, "
            f"Std={results[label]['std_dose']:.2f} Gy, "
            f"Uncertainty={results[label]['mean_uncertainty']:.2f} Gy"
        )

    return results


# ---------------------------------------------------------------------------
# DDF-compatible output
# ---------------------------------------------------------------------------
def write_dose_stats_file(all_results, data_dir, output_dir):
    """
    Write ``dose_stats_stable_diffusion.txt`` in the same format as
    the DDF method's ``dose_stats.txt``.
    """
    LABEL_MAP = {2: "rectum", 3: "bladder", 4: "prostate", 5: "femur heads"}
    output_file = f"{output_dir}/dose_stats_stable_diffusion.txt"

    with open(output_file, "w") as f:
        for pid in sorted(DOSE_FLIP_FLAGS.keys()):
            if pid not in all_results:
                continue

            f.write("%%%%%%%%%%%%%%%%%%%%\n")
            f.write(f"Patient_{pid}\n")

            # Planned dose
            dose_files = glob.glob(
                f"{data_dir}/DOSES/Patient_{pid}/Patient_{pid}_fraction_1_*.nii.gz"
            )
            if not dose_files:
                continue

            dose_raw = nib.load(dose_files[0]).get_fdata()
            flip = DOSE_FLIP_FLAGS.get(pid, 0)
            if flip:
                dose_raw = dose_raw[:, :, ::-1]
            dose_planned = dose_raw.swapaxes(2, 1).swapaxes(1, 0).swapaxes(2, 1)

            struct_path = f"{data_dir}/STRUCTURES/Patient_{pid}/Patient_{pid}_fraction_1_.nii.gz"
            struct_data = load_nifti_as_array(struct_path)

            f.write("\tPlanned doses means\n")
            for label in [2, 3, 4, 5]:
                mask = struct_data == label
                if mask.sum() > 0:
                    f.write(f"\t {LABEL_MAP[label]} {dose_planned[mask].mean()}\n")

            f.write("\tDose means from anatomical variants\n")
            patient_res = all_results[pid]
            for label in [2, 3, 4, 5]:
                if label in patient_res:
                    f.write(f"\t {LABEL_MAP[label]} {patient_res[label]['mean_dose']}\n")

    print(f"Dose stats written to: {output_file}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Dose uncertainty analysis")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--generated-dir", type=str, default=DEFAULT_GENERATED_DIR)
    parser.add_argument("--csd-results-dir", type=str, default=DEFAULT_CSD_RESULTS_DIR)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force-recalculate", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    patients_to_process = sorted(DOSE_FLIP_FLAGS.keys())
    all_results = {}

    for pid in patients_to_process:
        try:
            results = process_patient_dose_uncertainty(
                pid,
                args.data_dir,
                args.generated_dir,
                args.csd_results_dir,
                args.output_dir,
                force_recalculate=args.force_recalculate,
            )
            if results is not None:
                all_results[pid] = results
                with open(f"{args.output_dir}/patient_{pid}_results.json", "w") as f:
                    json.dump(results, f, indent=2)
        except Exception as e:
            print(f"ERROR for patient {pid}: {e}")
            import traceback
            traceback.print_exc()

    # Summary statistics
    summary = {}
    for label in STRUCTURE_LABELS:
        vals = {k: [] for k in ("mean_doses", "std_doses", "uncertainties")}
        for pr in all_results.values():
            if label in pr:
                vals["mean_doses"].append(pr[label]["mean_dose"])
                vals["std_doses"].append(pr[label]["std_dose"])
                vals["uncertainties"].append(pr[label]["mean_uncertainty"])
        if vals["mean_doses"]:
            summary[label] = {
                "structure_name": STRUCTURE_NAMES[label],
                "mean_dose_avg": float(np.mean(vals["mean_doses"])),
                "mean_dose_std": float(np.std(vals["mean_doses"])),
                "std_dose_avg": float(np.mean(vals["std_doses"])),
                "std_dose_std": float(np.std(vals["std_doses"])),
                "uncertainty_avg": float(np.mean(vals["uncertainties"])),
                "uncertainty_std": float(np.std(vals["uncertainties"])),
                "n_patients": len(vals["mean_doses"]),
            }

    with open(f"{args.output_dir}/summary_statistics.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print(f"\n{'=' * 80}\nDOSE UNCERTAINTY SUMMARY\n{'=' * 80}")
    print(f"{'Structure':<20} {'Mean Dose (Gy)':<20} {'Dose Std (Gy)':<20} {'Uncertainty (Gy)':<20}")
    print("-" * 80)
    for label in STRUCTURE_LABELS:
        if label in summary:
            s = summary[label]
            print(
                f"{STRUCTURE_NAMES[label]:<20} "
                f"{s['mean_dose_avg']:.2f} +/- {s['mean_dose_std']:.2f}         "
                f"{s['std_dose_avg']:.2f} +/- {s['std_dose_std']:.2f}         "
                f"{s['uncertainty_avg']:.2f} +/- {s['uncertainty_std']:.2f}"
            )

    # DDF-compatible output
    write_dose_stats_file(all_results, args.data_dir, args.output_dir)

    print("\nDose uncertainty analysis complete.")


if __name__ == "__main__":
    main()
