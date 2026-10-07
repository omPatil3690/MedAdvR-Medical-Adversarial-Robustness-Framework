import os
import torch
import matplotlib.pyplot as plt
import logging


# =========================
# NEW: Metric Function
# =========================
def compute_metrics(preds, targets, eps=1e-6):
    """
    So each image becomes one long vector of pixels:
    Image 1 → [pixel1, pixel2, ..., pixel16384]
    Image 2 → [pixel1, pixel2, ..., pixel16384]
    ...
    Image 8 → [...]

    Importantly, the batch dimension is preserved.
    3. Why flatten the images?
    Because now you can perform the classification-style calculations directly on every pixel.
    Originally:
    [B, 1, H, W]

    After flattening:
    [B, H×W]
    """
    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    TP = (preds * targets).sum(dim=1)
    FP = (preds * (1 - targets)).sum(dim=1)
    FN = ((1 - preds) * targets).sum(dim=1)
    TN = ((1 - preds) * (1 - targets)).sum(dim=1)
    """
            For segmentation, each pixel is effectively treated as a binary classification:
            Pixel:
            0 → background
            1 → tumor
            So you're essentially converting:
            image segmentation problem
            into:
            many pixel-level binary classification decisions
            for the purpose of calculating metrics.
    """
    dice = (2 * TP + eps) / (2 * TP + FP + FN + eps)
    iou = (TP + eps) / (TP + FP + FN + eps)
    precision = (TP + eps) / (TP + FP + eps)
    recall = (TP + eps) / (TP + FN + eps)

    return {
        "dice": dice.mean().item(),
        "iou": iou.mean().item(),
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
        "TP": TP.sum().item(),
        "FP": FP.sum().item(),
        "FN": FN.sum().item(),
        "TN": TN.sum().item(),
    }


# =========================
# MAIN EVALUATION
# =========================
def clean_evaluations(model, test_loader, device, save_dir="train_eval"):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    # 🔥 CHANGE: track all metrics
    metrics_sum = {
        "dice": 0,
        "iou": 0,
        "precision": 0,
        "recall": 0,
        "TP": 0,
        "FP": 0,
        "FN": 0,
        "TN": 0
    }

    count = 0

    with torch.no_grad():
        for i, (imgs, masks) in enumerate(test_loader):
            imgs, masks = imgs.to(device), masks.to(device)

            logits = model(imgs)
            preds = (torch.sigmoid(logits) > 0.5).float()

            # CHANGE: compute full metrics
            batch_metrics = compute_metrics(preds, masks)

            for k in metrics_sum:
                metrics_sum[k] += batch_metrics[k]

            count += 1

            # Save visuals
            if i < 5:
                _save_eval_plot(imgs, masks, preds, i, save_dir)

    # CHANGE: average metrics
    avg_metrics = {k: v / count for k, v in metrics_sum.items()}

    # =========================
    # SAVE EVERYTHING
    # =========================
    with open(os.path.join(save_dir, "summary.txt"), "w") as f:
        f.write("===== FINAL EVALUATION METRICS =====\n\n")

        f.write(f"Dice Score     : {avg_metrics['dice']:.6f}\n")
        f.write(f"IoU (Jaccard)  : {avg_metrics['iou']:.6f}\n")
        f.write(f"Precision      : {avg_metrics['precision']:.6f}\n")
        f.write(f"Recall         : {avg_metrics['recall']:.6f}\n\n")

        f.write("----- Pixel Statistics -----\n")
        f.write(f"True Positives : {avg_metrics['TP']:.2f}\n")
        f.write(f"False Positives: {avg_metrics['FP']:.2f}\n")
        f.write(f"False Negatives: {avg_metrics['FN']:.2f}\n")
        f.write(f"True Negatives : {avg_metrics['TN']:.2f}\n")

    # Logging
    logging.info(
        f"Dice: {avg_metrics['dice']:.4f} | "
        f"IoU: {avg_metrics['iou']:.4f} | "
        f"Precision: {avg_metrics['precision']:.4f} | "
        f"Recall: {avg_metrics['recall']:.4f}"
    )

    # 🔥 IMPORTANT: return only dice
    return avg_metrics["dice"]


# =========================
# Visualization
# =========================
def _save_eval_plot(imgs, masks, preds, batch_idx, save_dir):
    img = imgs[0].cpu().permute(1, 2, 0)
    mask = masks[0][0].cpu()
    pred = preds[0][0].cpu()

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("Input Image")
    plt.imshow(img)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Ground Truth")
    plt.imshow(mask, cmap='gray')
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Prediction")
    plt.imshow(pred, cmap='gray')
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"eval_batch_{batch_idx}.png"))
    plt.close()