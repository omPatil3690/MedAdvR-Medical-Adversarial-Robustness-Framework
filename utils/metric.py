import torch


def dice_loss(logits, targets, eps=1e-6):
    """
    Computes soft Dice loss for binary segmentation.
    Dice = 2 * |A ∩ B| / (|A| + |B|)
    Dice Loss = 1 - Dice
    """
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=(1, 2, 3))
    cardinality = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + eps) / (cardinality + eps)
    return 1.0 - dice.mean()


def dice_score(preds, targets, eps=1e-6):
    """
    Computes hard Dice score on binarized predictions.
    Dice = 2 * |A ∩ B| / (|A| + |B|)
    """
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    cardinality = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + eps) / (cardinality + eps)
    return dice.mean()


def iou_score(preds, targets, eps=1e-6):
    """
    Computes Jaccard Index / Intersection over Union (IoU) on binarized predictions.
    IoU = |A ∩ B| / |A ∪ B| = |A ∩ B| / (|A| + |B| - |A ∩ B|)
    """
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection
    iou = (intersection + eps) / (union + eps)
    return iou.mean()


def iou_loss(logits, targets, eps=1e-6):
    """
    Computes soft IoU / Jaccard loss for binary segmentation.
    IoU = |A ∩ B| / (|A| + |B| - |A ∩ B|)
    IoU Loss = 1 - IoU
    """
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection
    iou = (intersection + eps) / (union + eps)
    return 1.0 - iou.mean()