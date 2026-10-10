from torch.utils.data import DataLoader, TensorDataset , ConcatDataset
import torch
from .metric import dice_loss, dice_score , iou_loss , iou_score



# =========================
# ⚙️ Helpers
# =========================
def _resolve_loader(dataset):
    if isinstance(dataset, DataLoader):
        return dataset
    if hasattr(dataset, "get_loader"):
        return dataset.get_loader()
    if hasattr(dataset, "__iter__"):
        return dataset
    raise ValueError("Invalid dataset")


def _resolve_dataset(dataset):
    return dataset.dataset if hasattr(dataset, "dataset") else dataset


def _loader_settings(dataset):
    return (
        getattr(dataset, "batch_size", 16),
        getattr(dataset, "shuffle", True),
        getattr(dataset, "num_workers", 4),
        getattr(dataset, "pin_memory", False),
    )


def _make_loader(dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True):
    return DataLoader(dataset, batch_size, shuffle, num_workers, pin_memory)

#