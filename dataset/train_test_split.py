import torch
from torch.utils.data import random_split, DataLoader


def get_train_val_test_loaders(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, batch_size=16, seed=42):
    """
    Splits a dataset into Train (70%), Validation (15%), and Test (15%) sets.
    - Train: Model optimization.
    - Validation: Learning rate scheduling & model checkpointing (prevents data leakage).
    - Test: Strictly reserved for final unbiased benchmark evaluation.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Ratios must sum to 1.0"
    
    total_len = len(dataset)
    train_size = int(train_ratio * total_len)
    val_size = int(val_ratio * total_len)
    test_size = total_len - train_size - val_size

    train_ds, val_ds, test_ds = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )

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