"""
Diffeomorphic Demons registration between generated CT images and the
planning CT (fraction 1).

The transform maps the *generated* image onto the *fraction-1* coordinate
frame so that structures drawn on fraction 1 can be warped to approximate
the anatomy in the generated image.

Usage:
    python registration.py [--data-dir DIR] [--generated-dir DIR] [--output-dir DIR]
"""

import os
import glob
import argparse

import numpy as np
import nibabel as nib
import SimpleITK as sitk
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Image geometry constants (shared with the rest of the pipeline)
# ---------------------------------------------------------------------------
SPACING = (1.171875, 1.171875, 3.0)
ORIGIN = (0.0, 0.0, 0.0)
DIRECTION = (-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0)

STRUCTURE_LABELS = [2, 3, 4, 5]
STRUCTURE_NAMES = {
    2: "rectum",
    3: "bladder",
    4: "prostate",
    5: "femur_heads",
}

NUM_SAMPLES = 10

# Default paths
DEFAULT_DATA_DIR = "/net/tscratch/people/plgztabor/ROBUST_PLANNING/DATA"
DEFAULT_GENERATED_DIR = "/net/tscratch/people/plgpiotreksl/generated_samples/images_hu"
DEFAULT_OUTPUT_DIR = "/net/tscratch/people/plgpiotreksl/csd_verification_results_fixed"

# Test patient IDs
TEST_IDS = {
    "02", "03", "17", "24", "25", "27", "33",
    "42", "48", "57", "59", "68", "71", "76", "77",
}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def load_nifti_as_sitk(filepath, swap_axes=True):
    """Load a NIfTI file and return a SimpleITK image with correct geometry."""
    data = nib.load(filepath).get_fdata()
    if swap_axes:
        data = data.swapaxes(2, 1).swapaxes(1, 0).swapaxes(2, 1)

    sitk_img = sitk.GetImageFromArray(data.astype(np.float32))
    sitk_img.SetOrigin(ORIGIN)
    sitk_img.SetSpacing(SPACING)
    sitk_img.SetDirection(DIRECTION)
    return sitk_img


def extract_structure_mask(structure_data, label):
    """Return a binary SimpleITK mask for a single structure *label*."""
    mask = np.zeros(structure_data.shape, dtype=np.float32)
    mask[structure_data == label] = 1.0

    sitk_mask = sitk.GetImageFromArray(mask)
    sitk_mask.SetOrigin(ORIGIN)
    sitk_mask.SetSpacing(SPACING)
    sitk_mask.SetDirection(DIRECTION)
    return sitk_mask


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
def compute_registration_fixed(ct_frac1_image, generated_image, num_iterations=120):
    """
    Compute a Diffeomorphic Demons displacement field.

    Direction: fixed = CT_frac1, moving = generated.

    Returns
    -------
    displacement_transform : sitk.DisplacementFieldTransform
    disp_field : sitk.Image   (can be saved to disk)
    """
    demons = sitk.DiffeomorphicDemonsRegistrationFilter()
    demons.SetNumberOfIterations(num_iterations)
    demons.SetSmoothDisplacementField(True)
    demons.SetStandardDeviations(0.6)

    initial_field = sitk.Image(
        ct_frac1_image.GetWidth(),
        ct_frac1_image.GetHeight(),
        ct_frac1_image.GetDepth(),
        sitk.sitkVectorFloat64,
    )
    initial_field.CopyInformation(ct_frac1_image)

    result_disp = demons.Execute(ct_frac1_image, generated_image, initial_field)
    displacement_transform = sitk.DisplacementFieldTransform(result_disp)

    disp_field = sitk.TransformToDisplacementField(
        displacement_transform,
        sitk.sitkVectorFloat64,
        ct_frac1_image.GetSize(),
        ct_frac1_image.GetOrigin(),
        ct_frac1_image.GetSpacing(),
        ct_frac1_image.GetDirection(),
    )
    return displacement_transform, disp_field


def apply_transform_to_structure(structure_sitk, displacement_transform, reference_image):
    """Resample *structure_sitk* according to *displacement_transform*."""
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_image)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(displacement_transform)
    return resampler.Execute(structure_sitk)


# ---------------------------------------------------------------------------
# Per-patient processing
# ---------------------------------------------------------------------------
def process_patient(patient_id, data_dir, generated_dir, output_dir):
    """
    Register every generated sample for *patient_id*, warp fraction-1
    structures through the displacement field, and save all results.
    """
    print(f"\n{'=' * 60}")
    print(f"Processing patient: {patient_id}")
    print(f"{'=' * 60}")

    ct_path = f"{data_dir}/CT/Patient_{patient_id}/Patient_{patient_id}_fraction_1_.nii.gz"
    struct_path = f"{data_dir}/STRUCTURES/Patient_{patient_id}/Patient_{patient_id}_fraction_1_.nii.gz"

    if not os.path.exists(ct_path):
        print(f"ERROR: Missing CT — {ct_path}")
        return None
    if not os.path.exists(struct_path):
        print(f"ERROR: Missing structures — {struct_path}")
        return None

    sample_files = sorted(
        glob.glob(f"{generated_dir}/Patient_{patient_id}_sample_*_hu.nii.gz")
    )
    print(f"Found {len(sample_files)} generated samples")

    ct_frac1 = None
    struct_data = None
    structure_masks = {}
    warped_structures = {label: [] for label in STRUCTURE_LABELS}

    transforms_dir = f"{output_dir}/transforms"
    warped_dir = f"{output_dir}/warped_structures"
    os.makedirs(transforms_dir, exist_ok=True)
    os.makedirs(warped_dir, exist_ok=True)

    for sample_path in tqdm(sample_files, desc="Registering"):
        sample_name = os.path.basename(sample_path).replace(".nii.gz", "")

        # Check cache
        warped_paths = {
            label: f"{warped_dir}/warped_{sample_name}_Structure_{label}.nii.gz"
            for label in STRUCTURE_LABELS
        }
        if all(os.path.exists(p) for p in warped_paths.values()):
            for label in STRUCTURE_LABELS:
                arr = sitk.GetArrayFromImage(sitk.ReadImage(warped_paths[label]))
                warped_structures[label].append(arr)
            continue

        transform_path = f"{transforms_dir}/transform_{sample_name}.nii.gz"

        if os.path.exists(transform_path):
            disp_field = sitk.ReadImage(transform_path)
            displacement_transform = sitk.DisplacementFieldTransform(disp_field)
        else:
            if ct_frac1 is None:
                print("Loading fraction 1 CT ...")
                ct_frac1 = load_nifti_as_sitk(ct_path)
            generated_image = load_nifti_as_sitk(sample_path)
            displacement_transform, disp_field = compute_registration_fixed(
                ct_frac1, generated_image
            )
            sitk.WriteImage(disp_field, transform_path)

        # Lazy-load structures
        if struct_data is None:
            print("Loading fraction 1 segmentation ...")
            struct_data = (
                nib.load(struct_path)
                .get_fdata()
                .swapaxes(2, 1)
                .swapaxes(1, 0)
                .swapaxes(2, 1)
            )
            for label in STRUCTURE_LABELS:
                structure_masks[label] = extract_structure_mask(struct_data, label)

        if ct_frac1 is None:
            ct_frac1 = load_nifti_as_sitk(ct_path)

        for label in STRUCTURE_LABELS:
            warped = apply_transform_to_structure(
                structure_masks[label], displacement_transform, ct_frac1
            )
            warped_structures[label].append(sitk.GetArrayFromImage(warped))
            sitk.WriteImage(warped, warped_paths[label])

    return warped_structures


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Diffeomorphic Demons registration for CSD verification"
    )
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--generated-dir", type=str, default=DEFAULT_GENERATED_DIR)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    generated_files = glob.glob(
        f"{args.generated_dir}/Patient_*_sample_*_hu.nii.gz"
    )
    available_ids = {os.path.basename(f).split("_")[1] for f in generated_files}
    test_patient_ids = sorted(TEST_IDS & available_ids)
    print(f"Test patients: {test_patient_ids}")

    for pid in test_patient_ids:
        try:
            process_patient(pid, args.data_dir, args.generated_dir, args.output_dir)
        except Exception as e:
            print(f"ERROR for patient {pid}: {e}")
            import traceback
            traceback.print_exc()

    print("\nRegistration complete for all patients.")


if __name__ == "__main__":
    main()
