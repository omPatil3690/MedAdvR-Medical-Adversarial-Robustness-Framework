import torch
from torch.utils.data import random_split, DataLoader, Dataset
from .augmentation import SegmentationAugmentation


class AugmentedSubset(Dataset):
    """Wraps a Subset to apply stochastic data augmentation strictly during training."""
    def __init__(self, subset, augment=None):
        self.subset = subset
        self.augment = augment

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, mask = self.subset[idx]
        if self.augment is not None:
            img, mask = self.augment(img, mask)
        return img, mask


def get_train_val_test_loaders(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, batch_size=16, seed=42, augment_train=True, img_size=128):
    """
    Splits a dataset into Train (70%), Validation (15%), and Test (15%) sets.
    - Train: Model optimization (stochastically augmented on-the-fly if augment_train=True).
    - Validation: Learning rate scheduling & model checkpointing (strictly unaugmented clean scans).
    - Test: Strictly reserved for final unbiased benchmark evaluation (clean scans).
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Ratios must sum to 1.0"
    
    # If dataset has base underlying dataset, remove global augment so val/test remain pure
    base_ds = dataset.dataset if hasattr(dataset, "dataset") else dataset
    if hasattr(base_ds, "augment"):
        base_ds.augment = None

    total_len = len(base_ds)
    train_size = int(train_ratio * total_len)
    val_size = int(val_ratio * total_len)
    test_size = total_len - train_size - val_size

    train_ds, val_ds, test_ds = random_split(
        base_ds,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )

    if augment_train:
        train_ds = AugmentedSubset(train_ds, augment=SegmentationAugmentation(img_size))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=torch.cuda.is_available())

    return train_loader, val_loader, test_loader


def get_train_test_loaders(dataset, split_ratio=0.8, batch_size=16, seed=42):
    """
    Legacy 2-way split helper (backward compatibility).
    """
    train_size = int(split_ratio * len(dataset))
    test_size = len(dataset) - train_size

    train_ds, test_ds = random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=torch.cuda.is_available())

    return train_loader, test_loader