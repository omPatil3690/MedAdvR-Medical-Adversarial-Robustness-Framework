import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import logging
import matplotlib.pyplot as plt

from .eval import clean_evaluations

from utils.metric import iou_score, iou_loss , dice_loss , dice_score
from utils.train_helper import _resolve_dataset , _make_loader , _resolve_loader , _loader_settings

#  =========================
# TEST EPOCH
# =========================
def _test_epoch(model, loader, criterion, device, metric_type="dice"):
    model.eval()
    total_loss = 0.0
    total_metric_score = 0.0

    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)

            logits = model(imgs)
            
            # 1. Loss Calculation
            bce = criterion(logits, masks)
            # Choose loss based on metric_type
            m_loss = dice_loss(logits, masks) if metric_type == "dice" else iou_loss(logits, masks)
            loss = 0.5 * bce + 0.5 * m_loss
            total_loss += loss.item()

            # 2. Score Calculation
            preds = (torch.sigmoid(logits) > 0.35).float()
            if metric_type == "dice":
                total_metric_score += dice_score(preds, masks).item()
            else:
                total_metric_score += iou_score(preds, masks).item()

    return total_loss / len(loader), total_metric_score / len(loader)


# =========================
# TRAIN EPOCH
# =========================
def _train_epoch(model, loader, optimizer, criterion, device, metric_type="dice"):
    model.train()
    total_loss = 0.0
    total_metric_score = 0.0

    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        if masks.dim() == 3:
            masks = masks.unsqueeze(1)

        logits = model(imgs)
        
        # 1. Loss Calculation
        bce = criterion(logits, masks)
        # Fixed the 'mask' vs 'masks' typo and logic here
        m_loss = dice_loss(logits, masks) if metric_type == "dice" else iou_loss(logits, masks)
        loss = 0.5* bce + 0.5 * m_loss

        optimizer.zero_grad() # PyTorch accumulates gradients by default. Clearing them prevents gradients from previous batches from being unintentionally added to the current batch's gradients.
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

        # 2. Score Calculation
        with torch.no_grad():
            preds = (torch.sigmoid(logits) > 0.45).float()
            if metric_type == "dice":
                total_metric_score += dice_score(preds, masks).item()
            else:
                total_metric_score += iou_score(preds, masks).item()

    return total_loss / len(loader), total_metric_score / len(loader)
# =========================
# MAIN TRAIN
# =========================
def train_model_clean(model, train, test=None, device="cpu", epochs=5, lr=1e-3):
    metric_type = "iou"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(message)s',
        handlers=[
            logging.FileHandler("training.log"),  # 🔥 CHANGE
            logging.StreamHandler()
        ],
        force=True
    )

    train_loader = _resolve_loader(train)
    test_loader = _resolve_loader(test) if test else None

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=2, factor=0.5
    )
    
    pos_weight = torch.tensor([2.5]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    logging.info("Starting training...\n")  

    for epoch in range(epochs):
        train_loss, train_score = _train_epoch(model, train_loader, optimizer, criterion, device , metric_type=metric_type)

        if test_loader:
            test_loss, test_score = _test_epoch(model, test_loader, criterion, device , metric_type=metric_type)

            scheduler.step(test_loss)

            m_name = "Dice" if metric_type == "dice" else "IoU"
            logging.info(
                f"Epoch {epoch+1:02d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Test Loss: {test_loss:.4f} | "
                f"Train {m_name}: {train_score:.4f} | "
                f"Test {m_name}: {test_score:.4f}"
            )

    if test_loader:
        final_dice = clean_evaluations(model, test_loader, device, save_dir="train_eval")
        logging.info(f"\n[FINAL] Dice: {final_dice:.4f}")
        print(f"\n[FINAL] Dice: {final_dice:.4f}")

# =========================
# ADV TRAIN (unchanged)
# =========================
def train_model_adv(model, dataset, adv_buffer, device="cpu", epochs=3, lr=1e-3):
    clean_dataset = _resolve_dataset(dataset)

    if adv_buffer:
        adv_imgs, adv_masks = zip(*adv_buffer)
        adv_imgs = torch.stack(list(adv_imgs))
        adv_masks = torch.stack(list(adv_masks))
        adv_dataset = TensorDataset(adv_imgs, adv_masks)
        combined_dataset = ConcatDataset([clean_dataset, adv_dataset])
    else:
        combined_dataset = clean_dataset

    batch_size, shuffle, num_workers, pin_memory = _loader_settings(dataset)

    loader = _make_loader(combined_dataset, batch_size, shuffle, num_workers, pin_memory)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        loss, dice = _train_epoch(model, loader, optimizer, criterion, device)
        print(f"ADV Epoch {epoch+1}/{epochs} | Dice: {dice:.4f}")