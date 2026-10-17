"""
Multi-Task Brain Tumor Dataset & Stratified Splitter
====================================================
Loads paired (MRI Image, Segmentation Mask, Tumor Class Label) across:
  - Class 0: Glioma
  - Class 1: Meningioma
  - Class 2: Pituitary Tumor

Supports both:
  1. Nested class directories (e.g., ./data/glioma, ./data/meningioma, ./data/pituitary)
  2. Combined symlinked directory (e.g., ./data/all_tumors) by resolving symlink targets
     or matching filename index ranges.
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import torchvision.transforms as T

from .augmentation import SegmentationAugmentation

CLASS_NAMES = ["Glioma", "Meningioma", "Pituitary"]
CLASS_TO_IDX = {
    "glioma": 0,
    "meningioma": 1,
    "pituitary": 2,
    "pituitary_tumor": 2,
    "pituitary tumor": 2,
}


def _infer_class_from_path(img_path):
    """
    Determines the integer class label (0: Glioma, 1: Meningioma, 2: Pituitary)
    from the file path, its resolved symlink target, or sibling class folders.
    """
    real_path = os.path.realpath(img_path).lower()
    parent_folder = os.path.basename(os.path.dirname(real_path)).lower()

    for key, idx in CLASS_TO_IDX.items():
        if key in parent_folder or f"/{key}/" in real_path:
            return idx

    # Check sibling class directories (e.g. if img_path is inside ./data/all_tumors/enh_X.png)
    fname = os.path.basename(img_path)
    parent_dir = os.path.dirname(os.path.abspath(img_path))
    data_root = os.path.dirname(parent_dir)
    for folder_name, idx in [("glioma", 0), ("meningioma", 1), ("pituitary", 2), ("pituitary_tumor", 2)]:
        cand = os.path.join(data_root, folder_name, fname)
        if os.path.exists(cand):
            return idx
    
    raise ValueError(
    f"Cannot infer tumor class for image: {img_path}"
)
    


class MultiTaskTumorDataset(Dataset):
    """
    Returns (image, mask, class_label) for joint segmentation and classification.
      - image: FloatTensor [3, H, W] in [0, 1]
      - mask:  FloatTensor [1, H, W] in {0, 1}
      - label: LongTensor scalar in {0, 1, 2}
    """

    def __init__(self, root_dir, img_size=128, augment=None):
        self.root_dir = root_dir
        self.img_size = img_size
        self.augment = augment
        self.samples = []  # List of (img_path, mask_path, class_idx)

        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Dataset directory not found: {root_dir}")

        # Case 1: Direct images in root_dir (e.g. ./data/all_tumors or ./data/glioma)
        direct_imgs = sorted([
            f for f in os.listdir(root_dir)
            if f.endswith(".png") and not f.endswith("_mask.png")
        ])

        if direct_imgs:
            for img_name in direct_imgs:
                base = img_name[:-4]
                mask_name = f"{base}_mask.png"
                img_path = os.path.join(root_dir, img_name)
                mask_path = os.path.join(root_dir, mask_name)
                if os.path.exists(mask_path):
                    cls_idx = _infer_class_from_path(img_path)
                    self.samples.append((img_path, mask_path, cls_idx))
        else:
            # Case 2: Subdirectories inside root_dir (e.g. ./data)
            # Exclude 'all_tumors' if individual class folders exist to prevent duplicate counting
            subdirs = sorted([
                d for d in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, d)) and d != "all_tumors"
            ])
            for sub in subdirs:
                sub_path = os.path.join(root_dir, sub)
                for f in sorted(os.listdir(sub_path)):
                    if f.endswith(".png") and not f.endswith("_mask.png"):
                        base = f[:-4]
                        mask_name = f"{base}_mask.png"
                        img_path = os.path.join(sub_path, f)
                        mask_path = os.path.join(sub_path, mask_name)
                        if os.path.exists(mask_path):
                            cls_idx = _infer_class_from_path(img_path)
                            self.samples.append((img_path, mask_path, cls_idx))

        self.img_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.samples)

    def get_class_counts(self):
        counts = {0: 0, 1: 0, 2: 0}
        for _, _, c in self.samples:
            counts[c] = counts.get(c, 0) + 1
        return {CLASS_NAMES[k]: v for k, v in counts.items()}

    def __getitem__(self, idx):
        img_path, mask_path, cls_idx = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        image = self.img_transform(image)

        mask = Image.open(mask_path).convert("L")
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
        mask = np.array(mask)
        mask = (mask > 128).astype(np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0)

        if self.augment is not None:
            image, mask = self.augment(image, mask)

        label = torch.tensor(cls_idx, dtype=torch.long)
        return image, mask, label


class AugmentedMultiTaskSubset(Dataset):
    """Wraps a dataset subset so stochastic augmentations apply ONLY to the training split."""

    def __init__(self, subset, augment=None):
        self.subset = subset
        self.augment = augment

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, mask, label = self.subset[idx]
        if self.augment is not None:
            image, mask = self.augment(image, mask)
        return image, mask, label


def get_multitask_train_val_test_loaders(
    dataset_path,
    img_size=128,
    batch_size=16,
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42,
    augment_train=True
):
    """
    Creates a Stratified 3-Way Split (Train 70% / Val 15% / Test 15%) across
    all tumor classes so each split preserves identical pathology ratios.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Split ratios must sum to 1.0"

    base_dataset = MultiTaskTumorDataset(dataset_path, img_size=img_size, augment=None)

    # Group indices by class for stratified splitting
    class_indices = {0: [], 1: [], 2: []}
    for idx, (_, _, cls_idx) in enumerate(base_dataset.samples):
        class_indices.setdefault(cls_idx, []).append(idx)

    rng = random.Random(seed)
    train_indices, val_indices, test_indices = [], [], []

    for cls_idx, indices in class_indices.items():
        rng.shuffle(indices)
        n_total = len(indices)
        n_train = int(train_ratio * n_total)
        n_val = int(val_ratio * n_total)

        train_indices.extend(indices[:n_train])
        val_indices.extend(indices[n_train:n_train + n_val])
        test_indices.extend(indices[n_train + n_val:])

    # Shuffle combined indices deterministically
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)

    train_subset = Subset(base_dataset, train_indices)
    val_subset = Subset(base_dataset, val_indices)
    test_subset = Subset(base_dataset, test_indices)

    if augment_train:
        train_subset = AugmentedMultiTaskSubset(
            train_subset, augment=SegmentationAugmentation(img_size=img_size)
        )

    pin_mem = torch.cuda.is_available()
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, pin_memory=pin_mem)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, pin_memory=pin_mem)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, pin_memory=pin_mem)

    return train_loader, val_loader, test_loader, base_dataset
