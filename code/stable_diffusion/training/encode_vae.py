#!/usr/bin/env python3
import os
import sys
import glob
import argparse
from pathlib import Path
from datetime import datetime
import json

import torch
import numpy as np
import nibabel as nib
from tqdm import tqdm
from argparse import Namespace

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# MONAI imports
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRanged,
    Resized,
    Orientationd,
    EnsureTyped,
)


sys.path.append('/net/people/plgrid/plgpiotreksl/maisi/scripts')
from utils import define_instance
from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi


def setup_directories(latent_dir, input_dir):
    os.makedirs(latent_dir, exist_ok=True)
    os.makedirs(input_dir, exist_ok=True)
    print(f"Latent output directory: {latent_dir}")
    print(f"Input images directory: {input_dir}")


def load_model(device, model_path):

    print("Loading MAISI VAE model...")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU Memory before loading: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    
    args = Namespace()
    args.autoencoder_def = {
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
        "num_splits": 8,
        "dim_split": 0
    }
    
    autoencoder = define_instance(args, "autoencoder_def").to(device)
    
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        autoencoder.load_state_dict(state_dict)
        autoencoder.eval()
        
        print(f"Model loaded from: {model_path}")
        file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"Model size: {file_size_mb:.2f} MB")
        
        if torch.cuda.is_available():
            print(f"GPU Memory after loading: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    else:
        raise FileNotFoundError(f"Model not found at: {model_path}")
    
    transforms = Compose([
        LoadImaged(keys="image"),
        EnsureChannelFirstd(keys="image"),
        Orientationd(keys="image", axcodes="RAS"),
        EnsureTyped(keys="image", dtype=torch.float32),
        ScaleIntensityRanged(
            keys="image", a_min=-1000, a_max=1000, b_min=0, b_max=1, clip=True
        ),
        Resized(keys="image", spatial_size=[480, 480, 128], mode="trilinear"),
    ])
    
    print(f"Model loaded successfully on {device}")
    print(f"Model parameters: {sum(p.numel() for p in autoencoder.parameters()):,}")
    print(f"Target image size: (480, 480, 128)")
    print(f"Expected latent size: (120, 120, 32) (4x compression)")
    
    return autoencoder, transforms


def get_patient_files(data_dir):
    femur_files = sorted(glob.glob('/net/tscratch/people/plgztabor/DATA/FemurHeads/*.nii.gz'))
    patient_ids = sorted(set([os.path.basename(f).split('_')[1] for f in femur_files]))
    
    patient_files = {}
    total_fractions = 0
    
    for patient_id in patient_ids:
        patient_dir = os.path.join(data_dir, f"Patient_{patient_id}")
        
        if not os.path.exists(patient_dir):
            print(f"Warning: Directory not found for patient {patient_id}: {patient_dir}")
            continue
        
   
        fraction_pattern = os.path.join(patient_dir, f"Patient_{patient_id}_fraction_*.nii.gz")
        fraction_files = sorted(glob.glob(fraction_pattern))
        
        if fraction_files:
            patient_files[patient_id] = fraction_files
            total_fractions += len(fraction_files)
    
    print(f"\nFound {len(patient_files)} patients with {total_fractions} total fractions")
    return patient_files


def encode_image(model, transforms, img_path, device, verbose=False):
    try:

        img_nib = nib.load(img_path)
        original_shape = img_nib.shape
        original_affine = img_nib.affine
        
        if verbose:
            print(f"    Original shape: {original_shape}")
        
        data_dict = {"image": img_path}
        transformed_dict = transforms(data_dict)
        img_tensor = transformed_dict["image"]
        
        img_tensor = img_tensor.unsqueeze(0).to(device).to(torch.float16)
        
        if verbose:
            print(f"    Transformed tensor shape: {img_tensor.shape}")
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=True):
                latent_tuple = model.encode(img_tensor)
                if isinstance(latent_tuple, tuple):
                    latent = latent_tuple[0]
                else:
                    latent = latent_tuple
        
        if verbose:
            print(f"    Latent shape: {latent.shape}")
            print(f"    Latent range: [{latent.min().item():.4f}, {latent.max().item():.4f}]")

        latent_cpu = latent.cpu()
        transformed_cpu = img_tensor.cpu()
        
        # Cleanup GPU
        del img_tensor, latent, latent_tuple
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return latent_cpu, transformed_cpu, original_shape, original_affine
    
    except Exception as e:
        print(f"    ERROR encoding {img_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None, None


def save_latent(latent_tensor, output_path, metadata=None):
 
    data_to_save = {
        'latent': latent_tensor,
        'shape': tuple(latent_tensor.shape),
        'dtype': str(latent_tensor.dtype),
    }
    
    if metadata:
        data_to_save['metadata'] = metadata
    
    torch.save(data_to_save, output_path)


def save_transformed_image(transformed_tensor, output_path, original_affine=None):

    img_data = transformed_tensor.squeeze().cpu().float().numpy().astype(np.float32)
    
    affine = original_affine if original_affine is not None else np.eye(4)
    

    nib_img = nib.Nifti1Image(img_data, affine)
    nib.save(nib_img, output_path)


def process_patient(patient_id, fraction_files, model, transforms, device, 
                   latent_dir, input_dir, verbose=False):

    patient_latent_dir = os.path.join(latent_dir, f"Patient_{patient_id}")
    patient_input_dir = os.path.join(input_dir, f"Patient_{patient_id}")
    os.makedirs(patient_latent_dir, exist_ok=True)
    os.makedirs(patient_input_dir, exist_ok=True)
    
    stats = {
        'patient_id': patient_id,
        'total_fractions': len(fraction_files),
        'successful': 0,
        'failed': 0,
        'fractions': []
    }
    
    for fraction_path in fraction_files:
        fraction_name = os.path.basename(fraction_path).replace('.nii.gz', '')
        
        if verbose:
            print(f"  Processing: {fraction_name}")
        
        latent, transformed, orig_shape, orig_affine = encode_image(
            model, transforms, fraction_path, device, verbose=verbose
        )
        
        if latent is None:
            stats['failed'] += 1
            continue
        
        metadata = {
            'patient_id': patient_id,
            'fraction_name': fraction_name,
            'original_path': fraction_path,
            'original_shape': orig_shape,
            'transformed_shape': tuple(transformed.shape),
            'timestamp': datetime.now().isoformat(),
        }
        
        latent_path = os.path.join(patient_latent_dir, f"{fraction_name}.pt")
        save_latent(latent, latent_path, metadata)
        
        input_path = os.path.join(patient_input_dir, f"{fraction_name}.nii.gz")
        save_transformed_image(transformed, input_path, orig_affine)
        
        stats['successful'] += 1
        stats['fractions'].append({
            'name': fraction_name,
            'latent_path': latent_path,
            'input_path': input_path,
            'latent_shape': tuple(latent.shape),
            'original_shape': orig_shape,
        })
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Encode CT images to MAISI VAE latent space')
    parser.add_argument('--data-dir', type=str, 
                       default='/net/tscratch/people/plgztabor/DATA/CT',
                       help='Directory with patient CT data')
    parser.add_argument('--latent-dir', type=str,
                       default='/net/tscratch/people/plgpiotreksl/maisi_vae_latent',
                       help='Output directory for latent representations')
    parser.add_argument('--input-dir', type=str,
                       default='/net/tscratch/people/plgpiotreksl/maisi_vae_input',
                       help='Output directory for transformed input images')
    parser.add_argument('--model-path', type=str,
                       default='/net/people/plgrid/plgpiotreksl/jupyter/models/autoencoder_epoch273.pt',
                       help='Path to trained MAISI VAE model')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use (cuda:0, cpu, etc.)')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed progress information')
    parser.add_argument('--max-patients', type=int, default=None,
                       help='Maximum number of patients to process (for testing)')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    setup_directories(args.latent_dir, args.input_dir)
    
    model, transforms = load_model(device, args.model_path)
    
    patient_files = get_patient_files(args.data_dir)
    
    if args.max_patients:
        patient_ids = list(patient_files.keys())[:args.max_patients]
        patient_files = {pid: patient_files[pid] for pid in patient_ids}
        print(f"\nLimiting to {args.max_patients} patients for testing")
    
    print(f"\n{'='*80}")
    print("Starting encoding process...")
    print(f"{'='*80}\n")
    
    all_stats = []
    total_successful = 0
    total_failed = 0
    
    for patient_id in tqdm(sorted(patient_files.keys()), desc="Patients"):
        fraction_files = patient_files[patient_id]
        
        if args.verbose:
            print(f"\nProcessing Patient {patient_id} ({len(fraction_files)} fractions)")
        
        stats = process_patient(
            patient_id, fraction_files, model, transforms, device,
            args.latent_dir, args.input_dir, args.verbose
        )
        
        all_stats.append(stats)
        total_successful += stats['successful']
        total_failed += stats['failed']
        
        if not args.verbose:
            tqdm.write(f"Patient {patient_id}: {stats['successful']}/{stats['total_fractions']} successful")
    
    print(f"\n{'='*80}")
    print("ENCODING COMPLETE")
    print(f"{'='*80}")
    print(f"Total patients processed: {len(all_stats)}")
    print(f"Total fractions successful: {total_successful}")
    print(f"Total fractions failed: {total_failed}")
    print(f"Success rate: {100*total_successful/(total_successful+total_failed):.1f}%")
    
    summary_file = os.path.join(args.latent_dir, 'encoding_summary.json')
    summary = {
        'timestamp': datetime.now().isoformat(),
        'model': 'MAISI VAE',
        'model_path': args.model_path,
        'modality': 'CT',
        'device': str(device),
        'target_size': [480, 480, 128], ## rescaling all images to 1 consistent size (input images have various z dimension)
        'expected_latent_size': [120, 120, 32],
        'total_patients': len(all_stats),
        'total_successful': total_successful,
        'total_failed': total_failed,
        'success_rate': total_successful / (total_successful + total_failed) if (total_successful + total_failed) > 0 else 0,
        'patient_stats': all_stats,
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to: {summary_file}")
    print(f"Latent files saved in: {args.latent_dir}")
    print(f"Input files saved in: {args.input_dir}")
    print(f"\nAll latents should have shape: [1, 4, 120, 120, 32]")


if __name__ == '__main__':
    main()
