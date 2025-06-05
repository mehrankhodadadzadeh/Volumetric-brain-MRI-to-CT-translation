import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
from glob import glob
from monai.transforms import Compose, ToTensord, RandSpatialCropd  # + other augmentations


class PairedNiftiDataset(Dataset):
    """
    Custom Dataset for paired NIfTI images (e.g. MRI → CT).
    Applies 3D patch extraction and optional MONAI augmentations.
    """

    def __init__(self, root_dir, patch_size=(64, 64, 64), mode="train", augment=False):
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.mode = mode.lower()
        self.augment = augment
        self.patient_dirs = sorted(glob(os.path.join(root_dir, "*")))

        if self.augment and self.mode == "train":
            self.transform = Compose([
                RandSpatialCropd(keys=["image", "label", "mask"], roi_size=patch_size, random_size=False),
                # ... spatial + intensity augmentations ...
                ToTensord(keys=["image", "label", "mask"]),
            ])
        else:
            self.transform = Compose([
                RandSpatialCropd(keys=["image", "label", "mask"], roi_size=patch_size, random_size=False),
                ToTensord(keys=["image", "label", "mask"]),
            ])

    def __len__(self):
        return len(self.patient_dirs)

    def __getitem__(self, idx):
        # Load and normalize NIfTI files
        patient_dir = self.patient_dirs[idx]
        mri = nib.load(os.path.join(patient_dir, "mr.nii.gz")).get_fdata().astype(np.float32)
        ct  = nib.load(os.path.join(patient_dir, "ct.nii.gz")).get_fdata().astype(np.float32)
        mask = nib.load(os.path.join(patient_dir, "mask.nii.gz")).get_fdata().astype(np.float32)

        mri = (mri - mri.mean()) / (mri.std() + 1e-8)
        ct = (ct + 1000) / 3000.0

        mri = np.expand_dims(mri, 0)
        ct = np.expand_dims(ct, 0)
        mask = np.expand_dims(mask, 0)

        sample = self.transform({"image": mri, "label": ct, "mask": mask})
        return sample["image"], sample["label"], sample["mask"]


def get_dataloaders(train_dir, val_dir, test_dir, batch_size=1, patch_size=(64, 64, 64)):
    """
    Returns PyTorch DataLoaders for train/val/test splits using patch-based loading.
    """
    train_ds = PairedNiftiDataset(train_dir, patch_size, "train", augment=True)
    val_ds   = PairedNiftiDataset(val_dir,   patch_size, "val",   augment=False)
    test_ds  = PairedNiftiDataset(test_dir,  patch_size, "test",  augment=False)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=4),
    )
