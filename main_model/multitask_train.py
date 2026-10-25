"""
Multi-Task Defender Training & Balanced Adversarial Hardening
=============================================================
Implements:
  1. evaluate_multitask_epoch: Weighted per-sample evaluation of Segmentation Dice/IoU,
     Classification Accuracy, and Consistency Gate calibration.
  2. train_multitask_clean: Baseline pretraining with ReduceLROnPlateau on validation loss.
  3. train_multitask_adv: 50% Clean / 50% Adversarial balanced batch retraining that
     simultaneously hardens segmentation, pathology classification, and the
     Cross-Attention Consistency Gate (Clean -> 1.0, Adversarial Disconnect -> 0.0).
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import logging

from utils.metric import iou_score, dice_score
from utils.multitask_losses import multitask_defender_loss


def evaluate_multitask_epoch(model, loader, device):
    """Evaluates MultiTaskUNet on a validation or test DataLoader."""
    model.eval()
    total_loss = 0.0
    total_seg_loss = 0.0
    total_cls_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    total_cls_correct = 0
    total_cons_score = 0.0
    total_samples = 0

    with torch.no_grad():
        for imgs, masks, labels in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            if masks.dim() == 3:
                masks = masks.unsqueeze(1)
            masks = (masks > 0.5).float()

            b_size = imgs.size(0)
            seg_logits, cls_logits, cons_score, f_seg, f_cls = model(imgs, return_latents=True)

            loss, loss_dict = multitask_defender_loss(
                model, seg_logits, cls_logits, cons_score, f_seg, f_cls, masks, labels
            )
            total_loss += loss.item() * b_size
            total_seg_loss += loss_dict.get("seg_loss", 0.0) * b_size
            total_cls_loss += loss_dict.get("cls_loss", 0.0) * b_size

            preds_bin = (torch.sigmoid(seg_logits) > 0.5).float()
            total_iou += iou_score(preds_bin, masks).item() * b_size
            total_dice += dice_score(preds_bin, masks).item() * b_size

            pred_cls = torch.argmax(cls_logits, dim=1)
            total_cls_correct += (pred_cls == labels).sum().item()
            total_cons_score += cons_score.mean().item() * b_size
            total_samples += b_size

    if total_samples == 0:
        return {"loss": 0.0, "seg_loss": 0.0, "cls_loss": 0.0, "iou": 0.0, "dice": 0.0, "cls_acc": 0.0, "cons": 0.0}

    return {
        "loss": total_loss / total_samples,
        "seg_loss": total_seg_loss / total_samples,
        "cls_loss": total_cls_loss / total_samples,
        "iou": total_iou / total_samples,
        "dice": total_dice / total_samples,
        "cls_acc": total_cls_correct / total_samples,
        "cons": total_cons_score / total_samples,
    }


def _train_multitask_clean_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_seg_loss = 0.0
    total_cls_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    total_cls_correct = 0
    total_samples = 0

    for batch_idx, (imgs, masks, labels) in enumerate(loader):
        imgs = imgs.to(device)
        masks = masks.to(device)
        labels = labels.to(device)
        if masks.dim() == 3:
            masks = masks.unsqueeze(1)
        masks = (masks > 0.5).float()

        b_size = imgs.size(0)
        seg_logits, cls_logits, cons_score, f_seg, f_cls = model(imgs, return_latents=True)

        loss, loss_dict = multitask_defender_loss(
            model, seg_logits, cls_logits, cons_score, f_seg, f_cls, masks, labels
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * b_size
        total_seg_loss += loss_dict.get("seg_loss", 0.0) * b_size
        total_cls_loss += loss_dict.get("cls_loss", 0.0) * b_size

        with torch.no_grad():
            preds_bin = (torch.sigmoid(seg_logits) > 0.5).float()
            total_iou += iou_score(preds_bin, masks).item() * b_size
            total_dice += dice_score(preds_bin, masks).item() * b_size
            pred_cls = torch.argmax(cls_logits, dim=1)
            total_cls_correct += (pred_cls == labels).sum().item()

        total_samples += b_size

        if (batch_idx + 1) % 25 == 0 or (batch_idx + 1) == len(loader):
            c_loss = total_loss / total_samples
            c_seg = total_seg_loss / total_samples
            c_cls = total_cls_loss / total_samples
            c_dice = total_dice / total_samples
            c_acc = (total_cls_correct / total_samples) * 100.0
            print(f"  ├─ Batch [{batch_idx+1:2d}/{len(loader):2d}] | Loss: {c_loss:.4f} (Seg: {c_seg:.4f}, Cls: {c_cls:.4f}) | Dice: {c_dice:.4f} | Cls Acc: {c_acc:.1f}%")

    return {
        "loss": total_loss / max(1, total_samples),
        "seg_loss": total_seg_loss / max(1, total_samples),
        "cls_loss": total_cls_loss / max(1, total_samples),
        "iou": total_iou / max(1, total_samples),
        "dice": total_dice / max(1, total_samples),
        "cls_acc": total_cls_correct / max(1, total_samples),
    }


def train_multitask_clean(model, train_loader, val_loader=None, device="cpu", epochs=5, lr=1e-3):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.FileHandler("multitask_training.log"),
            logging.StreamHandler(),
        ],
        force=True,
    )

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=2, factor=0.5
    )

    logging.info("Starting Multi-Task Clean Pretraining (Segmentation + 3-Class Pathology)...\n")

    for epoch in range(epochs):
        tr_stats = _train_multitask_clean_epoch(model, train_loader, optimizer, device)

        if val_loader is not None:
            val_stats = evaluate_multitask_epoch(model, val_loader, device)
            scheduler.step(val_stats["loss"])
            logging.info(
                f"Pretrain Epoch {epoch+1:02d}/{epochs} | "
                f"Train Loss: {tr_stats['loss']:.4f} (Seg: {tr_stats['seg_loss']:.4f}, Cls: {tr_stats['cls_loss']:.4f}) | "
                f"Val Loss: {val_stats['loss']:.4f} (Seg: {val_stats['seg_loss']:.4f}, Cls: {val_stats['cls_loss']:.4f}) | "
                f"Train Dice: {tr_stats['dice']:.4f} | Val Dice: {val_stats['dice']:.4f} | "
                f"Train Cls Acc: {tr_stats['cls_acc']*100:.1f}% | Val Cls Acc: {val_stats['cls_acc']*100:.1f}%"
            )
        else:
            logging.info(
                f"Pretrain Epoch {epoch+1:02d}/{epochs} | "
                f"Train Loss: {tr_stats['loss']:.4f} (Seg: {tr_stats['seg_loss']:.4f}, Cls: {tr_stats['cls_loss']:.4f}) | "
                f"Train Dice: {tr_stats['dice']:.4f} | "
                f"Train Cls Acc: {tr_stats['cls_acc']*100:.1f}%"
            )


def train_multitask_adv(
    model,
    train_loader,
    adv_buffer,
    device="cpu",
    epochs=3,
    lr=1e-3,
    batch_size=16
):
    """
    Balanced 50% Clean / 50% Adversarial Retraining for MultiTaskUNet.
    Simultaneously trains:
      - Segmentation Decoder to recover true tumor boundaries under attack
      - Classification Head to resist pathology swapping
      - Cross-Attention Consistency Gate to output 1.0 on clean scans and 0.0 on attacked scans
    """
    model = model.to(device)
    clean_dataset = train_loader.dataset if hasattr(train_loader, "dataset") else train_loader

    half_batch = max(1, batch_size // 2)
    clean_sub_loader = DataLoader(
        clean_dataset,
        batch_size=half_batch,
        shuffle=True,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )

    if adv_buffer:
        # Unpack (x_adv, y_mask, y_cls, ...)
        adv_imgs = torch.stack([item[0] for item in adv_buffer])
        adv_masks = torch.stack([item[1] for item in adv_buffer])
        adv_labels = torch.stack([item[2].long() if isinstance(item[2], torch.Tensor) else torch.tensor(item[2], dtype=torch.long) for item in adv_buffer])
        adv_dataset = TensorDataset(adv_imgs, adv_masks, adv_labels)
        adv_sub_loader = DataLoader(
            adv_dataset,
            batch_size=half_batch,
            shuffle=True,
            drop_last=True,
            pin_memory=torch.cuda.is_available(),
        )
    else:
        adv_sub_loader = None

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_seg_loss = 0.0
        total_cls_loss = 0.0
        total_dice = 0.0
        total_cls_correct = 0
        total_gate_acc = 0
        total_samples = 0

        clean_iter = iter(clean_sub_loader)
        adv_iter = iter(adv_sub_loader) if adv_sub_loader else None

        num_batches = len(clean_sub_loader)
        for _ in range(num_batches):
            try:
                x_c, m_c, y_c = next(clean_iter)
            except StopIteration:
                clean_iter = iter(clean_sub_loader)
                x_c, m_c, y_c = next(clean_iter)

            if adv_iter is not None:
                try:
                    x_a, m_a, y_a = next(adv_iter)
                except StopIteration:
                    adv_iter = iter(adv_sub_loader)
                    x_a, m_a, y_a = next(adv_iter)

                x_batch = torch.cat([x_c, x_a], dim=0).to(device)
                m_batch = torch.cat([m_c, m_a], dim=0).to(device)
                y_batch = torch.cat([y_c, y_a], dim=0).to(device)

                # 1.0 for clean samples, 0.0 for adversarial semantic disconnect samples
                is_clean_flag = torch.cat([
                    torch.ones(x_c.size(0), device=device),
                    torch.zeros(x_a.size(0), device=device)
                ], dim=0)
            else:
                x_batch = x_c.to(device)
                m_batch = m_c.to(device)
                y_batch = y_c.to(device)
                is_clean_flag = torch.ones(x_c.size(0), device=device)

            if m_batch.dim() == 3:
                m_batch = m_batch.unsqueeze(1)
            m_batch = (m_batch > 0.5).float()

            b_size = x_batch.size(0)
            seg_logits, cls_logits, cons_score, f_seg, f_cls = model(x_batch, return_latents=True)

            loss, loss_dict = multitask_defender_loss(
                model=model,
                seg_logits=seg_logits,
                cls_logits=cls_logits,
                consistency_score=cons_score,
                f_seg=f_seg,
                f_cls=f_cls,
                masks=m_batch,
                labels=y_batch,
                is_clean_flag=is_clean_flag,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * b_size
            total_seg_loss += loss_dict.get("seg_loss", 0.0) * b_size
            total_cls_loss += loss_dict.get("cls_loss", 0.0) * b_size

            with torch.no_grad():
                preds_bin = (torch.sigmoid(seg_logits) > 0.5).float()
                total_dice += dice_score(preds_bin, m_batch).item() * b_size
                pred_cls = torch.argmax(cls_logits, dim=1)
                total_cls_correct += (pred_cls == y_batch).sum().item()
                gate_pred = (cons_score.squeeze(1) > 0.55).float()
                total_gate_acc += (gate_pred == is_clean_flag).sum().item()

            total_samples += b_size

            if (_ + 1) % 25 == 0 or (_ + 1) == num_batches:
                cur_l = total_loss / total_samples
                cur_s = total_seg_loss / total_samples
                cur_c = total_cls_loss / total_samples
                cur_d = total_dice / total_samples
                cur_acc = (total_cls_correct / total_samples) * 100.0
                cur_gate = (total_gate_acc / total_samples) * 100.0
                print(f"  ├─ Batch [{_+1:2d}/{num_batches:2d}] | Loss: {cur_l:.4f} (Seg: {cur_s:.4f}, Cls: {cur_c:.4f}) | Dice: {cur_d:.4f} | Cls Acc: {cur_acc:.1f}% | Gate Acc: {cur_gate:.1f}%")

        n_s = max(1, total_samples)
        print(
            f"ADV Hardening Epoch {epoch+1:02d}/{epochs} | "
            f"Loss: {total_loss/n_s:.4f} (Seg: {total_seg_loss/n_s:.4f}, Cls: {total_cls_loss/n_s:.4f}) | "
            f"Dice: {total_dice/n_s:.4f} | "
            f"Cls Acc: {(total_cls_correct/n_s)*100:.1f}% | "
            f"Triage Gate Acc: {(total_gate_acc/n_s)*100:.1f}%"
        )
