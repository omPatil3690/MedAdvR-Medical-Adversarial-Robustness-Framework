"""
Multi-Task Adversarial Sample & Annotated Card Exporter
=======================================================
Saves:
  1. Raw adversarial images & masks (`images/adv_XXXX.png`, `masks/mask_XXXX.png`)
  2. Annotated Multi-Task Sample Cards (`sample_cards/sample_XXXX.png`) containing:
     - Clean MRI Scan + True Pathology Class
     - Ground Truth Mask + True Class Badge
     - Attacked MRI + Perturbation Heatmap (10x)
     - Attacked Segmentation Overlay + Predicted Tumor Class, Confidence %,
       IoU/Dice Score, Semantic Disconnect Score, and Safety Triage Flag!
  3. Structured CSV Manifest (`sample_predictions.csv`) recording per-sample
     true class, predicted class, confidence, Dice, and disconnect score.
"""

import os
import csv
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms

from dataset.multitask_loader import CLASS_NAMES
from utils.metric import dice_score


def save_multitask_adv_samples(
    adv_samples,
    save_dir,
    model=None,
    device="cpu",
    max_cards=12,
    tau_audit=0.45
):
    """
    Saves adversarial samples to disk along with annotated Multi-Task Sample Cards
    and a CSV manifest containing class predictions, confidences, Dice, and triage flags.

    Args:
        adv_samples: List of tuples (x_adv, y_mask, y_cls, x_clean) or (x_adv, y_mask, y_cls)
        save_dir: Output directory (e.g., outputs_multitask/adv_samples/cycle_0)
        model: Optional trained MultiTaskUNet to annotate live predictions on saved samples
        device: Torch device
        max_cards: Number of rich visual diagnostic cards to render per cycle
        tau_audit: Disconnect threshold above which Safety Triage Flag is triggered
    """
    os.makedirs(save_dir, exist_ok=True)
    img_dir = os.path.join(save_dir, "images")
    mask_dir = os.path.join(save_dir, "masks")
    cards_dir = os.path.join(save_dir, "sample_cards")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(cards_dir, exist_ok=True)

    to_pil = transforms.ToPILImage()
    csv_path = os.path.join(save_dir, "sample_predictions.csv")

    if model is not None:
        model.eval()

    rows = []
    saved_count = 0

    for idx, item in enumerate(adv_samples):
        if len(item) == 4:
            x_adv, y_mask, y_cls, x_clean = item
        else:
            x_adv, y_mask, y_cls = item
            x_clean = x_adv

        cls_idx = int(y_cls.item()) if isinstance(y_cls, torch.Tensor) else int(y_cls)
        true_cls_name = CLASS_NAMES[cls_idx]

        # 1. Save raw adversarial image and ground truth mask
        img_pil = to_pil((x_adv.clamp(0, 1) * 255).byte())
        img_pil.save(os.path.join(img_dir, f"adv_{idx:04d}_{true_cls_name.lower()}.png"))

        m_tensor = y_mask.squeeze(0) if y_mask.dim() == 3 else y_mask
        mask_pil = to_pil((m_tensor.clamp(0, 1) * 255).byte().unsqueeze(0))
        mask_pil.save(os.path.join(mask_dir, f"mask_{idx:04d}_{true_cls_name.lower()}.png"))
        saved_count += 1

        # 2. Evaluate & export rich annotated cards for top samples
        if model is not None and idx < max_cards:
            with torch.no_grad():
                xa_dev = x_adv.unsqueeze(0).to(device)
                xc_dev = x_clean.unsqueeze(0).to(device)
                ym_dev = y_mask.unsqueeze(0).to(device)
                if ym_dev.dim() == 3:
                    ym_dev = ym_dev.unsqueeze(1)

                # Clean prediction
                c_seg_logits, c_cls_logits, c_cons = model(xc_dev)
                c_pred_mask = (torch.sigmoid(c_seg_logits) > 0.5).float()
                c_dice = dice_score(c_pred_mask, ym_dev).item()
                c_probs = F.softmax(c_cls_logits, dim=1)[0]
                c_pred_cls = int(torch.argmax(c_probs).item())
                c_conf = float(c_probs[c_pred_cls].item()) * 100.0

                # Attacked prediction
                a_seg_logits, a_cls_logits, a_cons = model(xa_dev)
                a_pred_mask = (torch.sigmoid(a_seg_logits) > 0.5).float()
                a_dice = dice_score(a_pred_mask, ym_dev).item()
                a_probs = F.softmax(a_cls_logits, dim=1)[0]
                a_pred_cls = int(torch.argmax(a_probs).item())
                a_conf = float(a_probs[a_pred_cls].item()) * 100.0

                disconnect_score = 1.0 - float(a_cons.item())
                triage_flagged = disconnect_score > tau_audit

                rows.append({
                    "sample_id": f"adv_{idx:04d}",
                    "true_class": true_cls_name,
                    "clean_pred_class": CLASS_NAMES[c_pred_cls],
                    "clean_conf_pct": f"{c_conf:.1f}",
                    "clean_dice": f"{c_dice:.4f}",
                    "adv_pred_class": CLASS_NAMES[a_pred_cls],
                    "adv_conf_pct": f"{a_conf:.1f}",
                    "adv_dice": f"{a_dice:.4f}",
                    "disconnect_score": f"{disconnect_score:.4f}",
                    "safety_triage": "REFER_TO_RADIOLOGIST" if triage_flagged else "PASS",
                })

                _render_sample_card(
                    x_clean=x_clean.cpu(),
                    x_adv=x_adv.cpu(),
                    y_mask=m_tensor.cpu(),
                    c_pred_mask=c_pred_mask[0, 0].cpu(),
                    a_pred_mask=a_pred_mask[0, 0].cpu(),
                    true_cls_name=true_cls_name,
                    c_cls_name=CLASS_NAMES[c_pred_cls],
                    c_conf=c_conf,
                    c_dice=c_dice,
                    a_cls_name=CLASS_NAMES[a_pred_cls],
                    a_conf=a_conf,
                    a_dice=a_dice,
                    disconnect_score=disconnect_score,
                    triage_flagged=triage_flagged,
                    save_path=os.path.join(cards_dir, f"sample_card_{idx:03d}_{true_cls_name.lower()}.png")
                )

    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    return saved_count


def _render_sample_card(
    x_clean,
    x_adv,
    y_mask,
    c_pred_mask,
    a_pred_mask,
    true_cls_name,
    c_cls_name,
    c_conf,
    c_dice,
    a_cls_name,
    a_conf,
    a_dice,
    disconnect_score,
    triage_flagged,
    save_path
):
    """Renders a 5-panel annotated sample card showing segmentation AND class predictions."""
    clean_np = x_clean.permute(1, 2, 0).numpy()
    adv_np = x_adv.permute(1, 2, 0).numpy()
    mask_np = y_mask.numpy()
    c_mask_np = c_pred_mask.numpy()
    a_mask_np = a_pred_mask.numpy()

    delta_np = np.abs(adv_np - clean_np).mean(axis=2) * 10.0

    # Clean overlay (Cyan)
    c_bg = clean_np.mean(axis=2)
    c_overlay = np.stack([c_bg, c_bg, c_bg], axis=-1)
    c_overlay[c_mask_np > 0.5] = [0.1, 0.9, 0.9]

    # Attacked overlay (Red)
    a_bg = adv_np.mean(axis=2)
    a_overlay = np.stack([a_bg, a_bg, a_bg], axis=-1)
    a_overlay[a_mask_np > 0.5] = [1.0, 0.2, 0.2]

    fig, axes = plt.subplots(1, 5, figsize=(20, 4.3))

    axes[0].set_title(f"1. Clean Scan\nTrue Class: {true_cls_name}", fontsize=10.5, fontweight="bold")
    axes[0].imshow(clean_np)
    axes[0].axis("off")

    axes[1].set_title(f"2. Ground Truth Mask\nPathology: {true_cls_name}", fontsize=10.5, fontweight="bold")
    axes[1].imshow(mask_np, cmap="gray", vmin=0, vmax=1)
    axes[1].axis("off")

    c_match_color = "teal" if c_cls_name == true_cls_name else "darkorange"
    axes[2].set_title(
        f"3. Clean Pred (Dice: {c_dice:.2f})\nPred: {c_cls_name} ({c_conf:.1f}%)",
        fontsize=10.5, fontweight="bold", color=c_match_color
    )
    axes[2].imshow(c_overlay)
    axes[2].axis("off")

    axes[3].set_title("4. Perturbation (10x)\nSemantic Disconnect Attack", fontsize=10.5, fontweight="bold")
    axes[3].imshow(delta_np, cmap="magma", vmin=0, vmax=0.5)
    axes[3].axis("off")

    triage_str = "[ALERT: REFER]" if triage_flagged else "[PASS: CONSISTENT]"
    axes[4].set_title(
        f"5. Attacked (Dice: {a_dice:.2f}) | {triage_str}\nPred: {a_cls_name} ({a_conf:.1f}%) | Disc: {disconnect_score:.2f}",
        fontsize=10, fontweight="bold", color="darkred"
    )
    axes[4].imshow(a_overlay)
    axes[4].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close()
