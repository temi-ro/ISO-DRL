import os
import glob
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
import random
from tqdm import tqdm


class BraTS2023_Slice_Dataset(Dataset):
    def __init__(self, root_dir, split='train', noise=0.0):
        self.root_dir = root_dir
        self.samples = []
        self.split = split
        self.noise = noise

        images_dir = os.path.join(root_dir, "imagesTr")
        labels_dir = os.path.join(root_dir, "labelsTr")
        
        print(f"Scanning {root_dir} for volumes...")
        image_files = glob.glob(os.path.join(images_dir, "*0003.nii.gz"))
        print(f"Found {len(image_files)} image files.")


        # Split patients into train/val based on patient ID
        patient_ids = set()
        for img_path in image_files:
            filename = os.path.basename(img_path)
            parts = filename.split('-')
            patient_id = parts[2] 
            trial = parts[3].split('_')[0]  
            patient_ids.add(f"{patient_id}-{trial}")

        # Need consistent ordering 
        patient_ids = sorted(list(patient_ids))

        # Reproducible shuffle
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
        elif split == 'full':
            selected_ids = patient_ids
        elif split == 'train_ratio':
            selected_ids = patient_ids[:int(0.8 * n)]
        elif split == 'val_ratio':
            selected_ids = patient_ids[int(0.8 * n):]
        elif split == 'val_tiny_ratio':
            selected_ids = patient_ids[int(0.8 * n):int(0.8 * n)+20]

        for img_path in tqdm(image_files):
            filename = os.path.basename(img_path)
            
            parts = filename.split('-')
            
            # Format: ['BraTS', 'GLI', '00001', '000_0003.nii.gz']
            patient_id = parts[2] 
            trial = parts[3].split('_')[0]  
            
            if f"{patient_id}-{trial}" not in selected_ids:
                continue
            
            label_name = f"BraTS-GLI-{patient_id}-{trial}.nii.gz"
            label_path = os.path.join(labels_dir, label_name)

            if not os.path.exists(label_path):
                continue

            seg_obj = nib.load(label_path)
            seg_data = seg_obj.get_fdata()

            n_slices = 135 
            for slice_idx in range(20, n_slices):
                # if slice_idx < 20 or slice_idx > 134:
                #     continue
                ratio_tumor = np.sum(seg_data[:, :, slice_idx]) / (seg_data.shape[0] * seg_data.shape[1])
                if "ratio" in split and (ratio_tumor < 0.03 or ratio_tumor > 0.035):  # Skip slices with very little tumor
                    continue
                elif ratio_tumor == 0.0:
                    continue
                
                    
                self.samples.append({
                    'image_path': img_path,
                    'label_path': label_path,
                    'slice_idx': slice_idx,
                    'patient_id': f"{patient_id}-{trial}"
                })
                

        print(f"Dataset loaded. Found {len(self.samples)} valid 2D slices.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        slice_idx = sample_info['slice_idx']
        
        # --- LOAD IMAGE ---
        img_obj = nib.load(sample_info['image_path'])
        img_slice = img_obj.dataobj[..., slice_idx] # Shape: [H, W]
        
        img_slice = np.array(img_slice).astype(np.float32)

        if self.noise > 0:
            rng = np.random.default_rng(idx)
            clean_std = np.std(img_slice[img_slice > 0]) if np.any(img_slice > 0) else 1.0
            actual_noise_std = self.noise * clean_std
            
            noise_mask = rng.normal(0, actual_noise_std, img_slice.shape)
            img_slice = img_slice + noise_mask

        # Expand dims to make it [1, H, W] for PyTorch
        image = np.expand_dims(img_slice, axis=0).astype(np.float32)

        # --- LOAD MASK ---
        seg_obj = nib.load(sample_info['label_path'])
        mask_slice = seg_obj.dataobj[..., slice_idx]
        mask = np.array(mask_slice).astype(np.float32)
        
        # --- PREPROCESSING ---
        mask[mask > 0] = 1.0
        
        if image.max() > 0:
            mean = image.mean()
            std = image.std()
            if std > 0:
                image = (image - mean) / std
        
        return torch.from_numpy(image), torch.from_numpy(mask), torch.from_numpy(mask)