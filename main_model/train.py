import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import logging
import os

from .eval import clean_evaluations
from utils.metric import iou_score, iou_loss, dice_loss, dice_score
from utils.train_helper import _resolve_dataset, _make_loader, _resolve_loader, _loader_settings


# =====================================================================
# 🧪 EVALUATION EPOCH (Weighted per-sample calculation)
# =====================================================================
def _test_epoch(model, loader, criterion, device, metric_type="iou"):
    model.eval()
    total_loss = 0.0
    total_metric_score = 0.0
    total_samples = 0

    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            if masks.dim() == 3:
                masks = masks.unsqueeze(1)
            masks = (masks > 0.5).float()

            b_size = imgs.size(0)
            logits = model(imgs)

            # 1. Loss Calculation
            bce = criterion(logits, masks)
            m_loss = dice_loss(logits, masks) if metric_type == "dice" else iou_loss(logits, masks)
            loss = 0.5 * bce + 0.5 * m_loss
            total_loss += loss.item() * b_size

            # 2. Score Calculation (Weighted per sample)
            preds = (torch.sigmoid(logits) > 0.5).float()
            if metric_type == "dice":
                total_metric_score += dice_score(preds, masks).item() * b_size
            else:
                total_metric_score += iou_score(preds, masks).item() * b_size

            total_samples += b_size

    if total_samples == 0:
        return 0.0, 0.0
    return total_loss / total_samples, total_metric_score / total_samples


# =====================================================================
# 🏋️ TRAIN EPOCH (Weighted per-sample calculation)
# =====================================================================
def _train_epoch(model, loader, optimizer, criterion, device, metric_type="iou"):
    model.train()
    total_loss = 0.0
    total_metric_score = 0.0
    total_samples = 0

    for imgs, masks in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)
        if masks.dim() == 3:
            masks = masks.unsqueeze(1)
        masks = (masks > 0.5).float()

        b_size = imgs.size(0)
        logits = model(imgs)

        # 1. Loss Calculation
        bce = criterion(logits, masks)
        m_loss = dice_loss(logits, masks) if metric_type == "dice" else iou_loss(logits, masks)
        loss = 0.5 * bce + 0.5 * m_loss

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for numerical stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * b_size

        # 2. Score Calculation (Weighted per sample)
        with torch.no_grad():
            preds = (torch.sigmoid(logits) > 0.5).float()
            if metric_type == "dice":
                total_metric_score += dice_score(preds, masks).item() * b_size
            else:
                total_metric_score += iou_score(preds, masks).item() * b_size

        total_samples += b_size

    if total_samples == 0:
        return 0.0, 0.0
    return total_loss / total_samples, total_metric_score / total_samples


# =====================================================================
# 🚀 CLEAN PRETRAINING (Uses Validation set for Scheduler)
# =====================================================================
def train_model_clean(model, train, val=None, device="cpu", epochs=5, lr=1e-3):
    metric_type = "iou"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(message)s',
        handlers=[
            logging.FileHandler("training.log"),
            logging.StreamHandler()
        ],
        force=True
    )

    model = model.to(device)
    train_loader = _resolve_loader(train)
    val_loader = _resolve_loader(val) if val else None

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=2, factor=0.5
    )

    pos_weight = torch.tensor([2.5], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    logging.info("Starting clean baseline pretraining...\n")

    for epoch in range(epochs):
        train_loss, train_score = _train_epoch(
            model, train_loader, optimizer, criterion, device, metric_type=metric_type
        )

        m_name = "Dice" if metric_type == "dice" else "IoU"
        if val_loader:
            val_loss, val_score = _test_epoch(
                model, val_loader, criterion, device, metric_type=metric_type
            )
            scheduler.step(val_loss)

            logging.info(
                f"Pretrain Epoch {epoch+1:02d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Train {m_name}: {train_score:.4f} | "
                f"Val {m_name}: {val_score:.4f}"
            )
        else:
            logging.info(
                f"Pretrain Epoch {epoch+1:02d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train {m_name}: {train_score:.4f}"
            )


# =====================================================================
# ⚔️ BALANCED ADVERSARIAL RETRAINING (50% Clean / 50% Adv per Batch)
# =====================================================================
def train_model_adv(model, dataset, adv_buffer, device="cpu", epochs=3, lr=1e-3, batch_size=16):
    model = model.to(device)
    clean_dataset = _resolve_dataset(dataset)

    # Setup clean loader
    clean_loader = DataLoader(
        clean_dataset,
        batch_size=max(1, batch_size // 2),
        shuffle=True,
        drop_last=True,
        pin_memory=torch.cuda.is_available()
    )

    # Setup adv loader
    if adv_buffer:
        adv_imgs, adv_masks = zip(*adv_buffer)
        adv_imgs = torch.stack(list(adv_imgs))
        adv_masks = torch.stack(list(adv_masks))
        adv_dataset = TensorDataset(adv_imgs, adv_masks)
        adv_loader = DataLoader(
            adv_dataset,
            batch_size=max(1, batch_size // 2),
            shuffle=True,
            drop_last=True,
            pin_memory=torch.cuda.is_available()
        )
    else:
        adv_loader = None

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    pos_weight = torch.tensor([2.5], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_iou = 0.0
        total_samples = 0

        clean_iter = iter(clean_loader)
        adv_iter = iter(adv_loader) if adv_loader else None

        num_batches = len(clean_loader)
        for _ in range(num_batches):
            try:
                x_clean, y_clean = next(clean_iter)
            except StopIteration:
                clean_iter = iter(clean_loader)
                x_clean, y_clean = next(clean_iter)

            if adv_iter:
                try:
                    x_adv, y_adv = next(adv_iter)
                except StopIteration:
                    adv_iter = iter(adv_loader)
                    x_adv, y_adv = next(adv_iter)

                # Balanced fusion: 50% clean + 50% adversarial in every batch
                x_batch = torch.cat([x_clean, x_adv], dim=0).to(device)
                y_batch = torch.cat([y_clean, y_adv], dim=0).to(device)
            else:
                x_batch = x_clean.to(device)
                y_batch = y_clean.to(device)

            if y_batch.dim() == 3:
                y_batch = y_batch.unsqueeze(1)
            y_batch = (y_batch > 0.5).float()

            b_size = x_batch.size(0)
            logits = model(x_batch)

            bce = criterion(logits, y_batch)
            iou_l = iou_loss(logits, y_batch)
            loss = 0.5 * bce + 0.5 * iou_l

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * b_size
            with torch.no_grad():
                preds = (torch.sigmoid(logits) > 0.5).float()
                total_iou += iou_score(preds, y_batch).item() * b_size
            total_samples += b_size

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        avg_iou = total_iou / total_samples if total_samples > 0 else 0.0
        print(f"ADV Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | IoU: {avg_iou:.4f}")