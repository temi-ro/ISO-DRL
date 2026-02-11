import os
import glob
import numpy as np
import nibabel as nib
import random
import json
from tqdm import tqdm

# Create a new dataset for nnU-Net from BraTS2023 with specific slice selection criteria
# The new dataset will be named Dataset002_BraTS2023
# It will contain 2D slices from the original 3D volumes
# Selection criteria: slices where the ratio of tumor pixels to total pixels is between 0.03 and 0.035

# Dataset002_BraTS2023: Slices with tumor ratio between 0.03 and 0.035
# Dataset003_BraTS2023: Slices with any tumor presence (ratio > 0)
# Dataset004_BraTS2023: Slices with tumor ratio between 0.03 and 0.035 (val_ratio) + Gaussian noise added (sigma=0.1)
# Dataset005_BraTS2023: Slices with tumor ratio between 0.03 and 0.035 (val_ratio) + NO GAUSSIAN NOISE
# Dataset006_BraTS2023: Slices with tumor ratio between 0.03 and 0.035 (val) + Gaussian noise added (sigma=0.01)
# Dataset007_BraTS2023: Slices with tumor ratio between 0.03 and 0.035 (val) + Gaussian noise added (sigma=0.05)
# Dataset009_BraTS2023: Randomly selected 100 slices from the full dataset (no filtering by tumor ratio)


# --- CONFIGURATION ---
split = 'val' 

NEW_DATASET_ID = "009"
NEW_DATASET_NAME = f"Dataset{int(NEW_DATASET_ID):03d}_BraTS2023"
NOISE = 0.05

source_root_dir = "/vol/gpudata/trm25-iso/braTS2023/Dataset001_BraTS2023"
output_base_dir = "/vol/gpudata/trm25-iso/braTS2023"

target_imagesTr = os.path.join(output_base_dir, NEW_DATASET_NAME, "imagesTr")
target_labelsTr = os.path.join(output_base_dir, NEW_DATASET_NAME, "labelsTr")

os.makedirs(target_imagesTr, exist_ok=True)
os.makedirs(target_labelsTr, exist_ok=True)

images_dir = os.path.join(source_root_dir, "imagesTr")
labels_dir = os.path.join(source_root_dir, "labelsTr")

print(f"Scanning {source_root_dir}...")
image_files = glob.glob(os.path.join(images_dir, "*0003.nii.gz"))


patient_ids = set()
for img_path in image_files:
    filename = os.path.basename(img_path)
    parts = filename.split('-')
    patient_id = parts[2]
    trial = parts[3].split('_')[0]
    patient_ids.add(f"{patient_id}-{trial}")

patient_ids = sorted(list(patient_ids))
random.Random(1234).shuffle(patient_ids) 

n = len(patient_ids)
selected_ids = []

if split == 'train':
    selected_ids = patient_ids[:int(0.8 * n)]
elif split == 'train_small':
    selected_ids = patient_ids[:int(0.3 * n)]
elif split == 'val':
    selected_ids = patient_ids[int(0.8 * n):]
elif split == 'val_small':
    selected_ids = patient_ids[int(0.8 * n):int(0.9 * n)]
elif split == 'val_tiny':
    selected_ids = patient_ids[int(0.8 * n):int(0.8 * n)+20]
elif split == 'full':
    selected_ids = patient_ids

print(f"Selected {len(selected_ids)} patients for extraction.")

valid_slices_count = 0

for img_path in tqdm(image_files):
    filename = os.path.basename(img_path)
    parts = filename.split('-')
    patient_id = parts[2]
    trial = parts[3].split('_')[0]
    
    if f"{patient_id}-{trial}" not in selected_ids:
        continue

    label_name = f"BraTS-GLI-{patient_id}-{trial}.nii.gz"
    label_path = os.path.join(labels_dir, label_name)

    if not os.path.exists(label_path):
        continue

    # Load data
    img_obj = nib.load(img_path)
    seg_obj = nib.load(label_path)
    
    img_data = img_obj.get_fdata()
    seg_data = seg_obj.get_fdata()
    affine = img_obj.affine

    n_slices = img_data.shape[2]
    
    # Slice Loop
    for slice_idx in range(20, 135): 
        if slice_idx >= n_slices: continue

        slice_seg = seg_data[:, :, slice_idx]
        ratio_tumor = np.sum(slice_seg) / (slice_seg.shape[0] * slice_seg.shape[1])
        
        # Filter by Content
        if ratio_tumor < 0.03 or ratio_tumor > 0.035:
            continue
        # if ratio_tumor == 0:
            # continue

        unique_id = f"BraTS_{patient_id}_{trial}_slice{slice_idx:03d}"
        
        slice_img = img_data[:, :, slice_idx]

        if NOISE > 0:
            rng = np.random.default_rng()
            
            brain_mask = slice_img > 0
            clean_std = np.std(slice_img[brain_mask]) if np.any(brain_mask) else 1.0
            
            actual_noise_std = NOISE * clean_std
            noise = rng.normal(0, actual_noise_std, slice_img.shape)
            slice_img = slice_img + noise

        slice_img = slice_img[:, :, np.newaxis] # Add Z dim of 1
        
        slice_label = slice_seg[:, :, np.newaxis]
        slice_label[slice_label > 0] = 1 

        # Save 
        new_img_obj = nib.Nifti1Image(slice_img, affine)
        new_seg_obj = nib.Nifti1Image(slice_label, affine)
        
        nib.save(new_img_obj, os.path.join(target_imagesTr, f"{unique_id}_0000.nii.gz"))
        nib.save(new_seg_obj, os.path.join(target_labelsTr, f"{unique_id}.nii.gz"))
        
        valid_slices_count += 1

print(f"Done! Created {valid_slices_count} slice-files in {target_imagesTr}")