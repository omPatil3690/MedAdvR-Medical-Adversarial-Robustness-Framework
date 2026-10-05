import torch
from torch.utils.data import random_split, DataLoader

def get_train_test_loaders(dataset, split_ratio=0.7, batch_size=16, seed=42):
    """
    Splits a dataset into train and test sets and returns DataLoaders.
    """
    # Calculate lengths
    train_size = int(split_ratio * len(dataset))
    test_size = len(dataset) - train_size
    
    # Split the dataset
    train_dataset, test_dataset = random_split(
        dataset, 
        [train_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    # Create Loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader