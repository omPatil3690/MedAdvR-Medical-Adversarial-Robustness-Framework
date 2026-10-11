"""
MedAdvR Testing & Robustness Benchmark Suite
============================================
Evaluates a trained segmentation model under clean and adversarial conditions.
Produces high-contrast clinical diagnostic overlays and robustness reports.
"""

import os
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np

from dataset.loader import DatasetLoader
from dataset.train_test_split import get_train_val_test_loaders
from main_model.model import init_model
from generator.model import init_generator
from utils.metric import dice_score, iou_score
from main_model.eval import compute_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="MedAdvR Testing & Benchmark Suite")
    parser.add_argument("--dataset_path", type=str, default="./data/glioma", help="Path to dataset")
    parser.add_argument("--dataset_type", type=str, default="tumor", help="Dataset type: tumor, extracted, dicom, nifti")
    parser.add_argument("--model_type", type=str, default="Unet", help="Model architecture")
    parser.add_argument("--model_path", type=str, default="outputs/model_final.pth", help="Path to trained model weights")
    parser.add_argument("--gen_path", type=str, default="outputs/generator_final.pth", help="Path to trained generator weights")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cuda, mps, cpu")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--save_dir", type=str, default="outputs/test_results")
    parser.add_argument("--num_visual_samples", type=int, default=8, help="Number of visual sample figures to export")
    return parser.parse_args()


def run_benchmark(model, generator, test_loader, device, save_dir, num_visual_samples=8):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    if generator:
        generator.eval()

    clean_metrics = {"dice": 0, "iou": 0, "precision": 0, "recall": 0}
    adv_metrics = {"dice": 0, "iou": 0, "precision": 0, "recall": 0}
    total_samples = 0
    visual_saved = 0

    with torch.no_grad():
        for batch_idx, (imgs, masks) in enumerate(test_loader):
            imgs = imgs.to(device)
            masks = masks.to(device)
            if masks.dim() == 3:
                masks = masks.unsqueeze(1)
            masks = (masks > 0.5).float()

            b_size = imgs.size(0)

            # 1. Clean Inference
            clean_logits = model(imgs)
            clean_preds = (torch.sigmoid(clean_logits) > 0.5).float()
            c_metrics = compute_metrics(clean_preds, masks)
            for k in clean_metrics:
                clean_metrics[k] += c_metrics[k] * b_size

            # 2. Adversarial Inference
            if generator:
                perturb = generator(imgs)
                adv_imgs = torch.clamp(imgs + perturb, 0.0, 1.0)
            else:
                adv_imgs = torch.clamp(imgs + torch.randn_like(imgs) * 0.05, 0.0, 1.0)

            adv_logits = model(adv_imgs)
            adv_preds = (torch.sigmoid(adv_logits) > 0.5).float()
            a_metrics = compute_metrics(adv_preds, masks)
            for k in adv_metrics:
                adv_metrics[k] += a_metrics[k] * b_size

            total_samples += b_size

            # 3. Save Visual Diagnostic Comparison Figures
            if visual_saved < num_visual_samples:
                for i in range(min(b_size, num_visual_samples - visual_saved)):
                    c_dice = dice_score(clean_preds[i:i+1], masks[i:i+1]).item()
                    a_dice = dice_score(adv_preds[i:i+1], masks[i:i+1]).item()
                    
                    _save_diagnostic_plot(
                        img=imgs[i].cpu(),
                        mask=masks[i][0].cpu(),
                        clean_pred=clean_preds[i][0].cpu(),
                        adv_img=adv_imgs[i].cpu(),
                        adv_pred=adv_preds[i][0].cpu(),
                        clean_dice=c_dice,
                        adv_dice=a_dice,
                        sample_idx=visual_saved,
                        save_dir=save_dir
                    )
                    visual_saved += 1

    # Compute Weighted Averages
    for k in clean_metrics:
        clean_metrics[k] /= total_samples
        adv_metrics[k] /= total_samples

    robustness_retention = (adv_metrics["dice"] / (clean_metrics["dice"] + 1e-6)) * 100.0

    # Print Report
    print("\n" + "=" * 65)
    print("📊 MEDADVR COMPREHENSIVE TEST & ROBUSTNESS REPORT")
    print("=" * 65)
    print(f"Total Test Samples Evaluated: {total_samples}")
    print("-" * 65)
    print(f"{'Metric':<18} | {'Clean Performance':<18} | {'Under Adversarial Attack':<18}")
    print("-" * 65)
    print(f"{'Dice Score':<18} | {clean_metrics['dice']:<18.4f} | {adv_metrics['dice']:<18.4f}")
    print(f"{'IoU (Jaccard)':<18} | {clean_metrics['iou']:<18.4f} | {adv_metrics['iou']:<18.4f}")
    print(f"{'Precision':<18} | {clean_metrics['precision']:<18.4f} | {adv_metrics['precision']:<18.4f}")
    print(f"{'Recall':<18} | {clean_metrics['recall']:<18.4f} | {adv_metrics['recall']:<18.4f}")
    print("-" * 65)
    print(f"🛡️  ROBUSTNESS RETENTION SCORE: {robustness_retention:.2f}% (Adv Dice / Clean Dice)")
    print("=" * 65)
    print(f"💾 Visual Diagnostic plots saved to: {save_dir}/")

    # Save text scorecard
    scorecard_path = os.path.join(save_dir, "test_scorecard.txt")
    with open(scorecard_path, "w") as f:
        f.write("===== MEDADVR ROBUSTNESS SCORECARD =====\n\n")
        f.write(f"Total Samples Evaluated: {total_samples}\n\n")
        f.write(f"Clean Dice Score       : {clean_metrics['dice']:.4f}\n")
        f.write(f"Attacked Dice Score    : {adv_metrics['dice']:.4f}\n")
        f.write(f"Clean IoU (Jaccard)    : {clean_metrics['iou']:.4f}\n")
        f.write(f"Attacked IoU (Jaccard) : {adv_metrics['iou']:.4f}\n")
        f.write(f"Robustness Retention   : {robustness_retention:.2f}%\n")


def _save_diagnostic_plot(img, mask, clean_pred, adv_img, adv_pred, clean_dice, adv_dice, sample_idx, save_dir):
    """
    Saves an enhanced 5-panel clinical diagnostic comparison:
    1. Clean MRI Scan
    2. Ground Truth Mask (Crisp Black/White)
    3. Clean Prediction with Overlay (Green/Cyan)
    4. Attack Perturbation Map (10x amplified heatmap)
    5. Attacked Prediction with Overlay (Red/Orange)
    """
    img_np = img.permute(1, 2, 0).numpy()
    adv_np = adv_img.permute(1, 2, 0).numpy()
    mask_np = mask.numpy()
    clean_pred_np = clean_pred.numpy()
    adv_pred_np = adv_pred.numpy()

    # Perturbation delta magnified 10x
    delta_np = np.abs(adv_np - img_np).mean(axis=2) * 10.0

    # Overlays on grayscale background
    gray_bg = img_np.mean(axis=2)
    
    # Overlay: Clean Prediction (Cyan/Green overlay on MRI)
    clean_overlay = np.stack([gray_bg, gray_bg, gray_bg], axis=-1)
    clean_overlay[clean_pred_np > 0.5] = [0.1, 0.9, 0.9]  # Cyan overlay

    # Overlay: Attacked Prediction (Red/Orange overlay on MRI)
    adv_overlay = np.stack([adv_np.mean(axis=2)] * 3, axis=-1)
    adv_overlay[adv_pred_np > 0.5] = [1.0, 0.2, 0.2]  # Red overlay

    fig, axes = plt.subplots(1, 5, figsize=(19, 4))

    # 1. Clean MRI Scan
    axes[0].set_title("1. Clean Scan", fontsize=11, fontweight='bold')
    axes[0].imshow(img_np)
    axes[0].axis("off")

    # 2. Ground Truth Mask
    axes[1].set_title("2. Ground Truth Mask", fontsize=11, fontweight='bold')
    axes[1].imshow(mask_np, cmap="gray", vmin=0, vmax=1)
    axes[1].axis("off")

    # 3. Clean Prediction Overlay
    axes[2].set_title(f"3. Clean Pred (Dice: {clean_dice:.2f})", fontsize=11, fontweight='bold', color='teal')
    axes[2].imshow(clean_overlay)
    axes[2].axis("off")

    # 4. Attack Perturbation Heatmap
    axes[3].set_title("4. Perturbation (10x)", fontsize=11, fontweight='bold')
    im_p = axes[3].imshow(delta_np, cmap="magma", vmin=0, vmax=0.5)
    axes[3].axis("off")

    # 5. Attacked Prediction Overlay
    axes[4].set_title(f"5. Attacked Pred (Dice: {adv_dice:.2f})", fontsize=11, fontweight='bold', color='darkred')
    axes[4].imshow(adv_overlay)
    axes[4].axis("off")

    plt.tight_layout()
    plot_path = os.path.join(save_dir, f"diagnostic_sample_{sample_idx:03d}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    args = parse_args()

    # Device Setup
    if args.device == "auto":
        if torch.cuda.is_available():
            args.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"
    device = torch.device(args.device)

    print(f"[INFO] Running MedAdvR Benchmark on Device: {device}")

    # Dataset Loader (Consistent 3-way split: 70% Train / 15% Val / 15% Test)
    dataset = DatasetLoader(
        args.dataset_path,
        dataset_type=args.dataset_type,
        img_size=args.img_size,
        batch_size=args.batch_size
    )
    _, _, test_loader = get_train_val_test_loaders(
        dataset,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        batch_size=args.batch_size,
        seed=42
    )

    # Initialize Model & Generator
    model = init_model(args.model_type).to(device)
    if os.path.exists(args.model_path):
        print(f"[INFO] Loading trained model weights from: {args.model_path}")
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    else:
        print(f"[WARNING] Model checkpoint '{args.model_path}' not found. Using initialized weights.")

    generator = init_generator("Unet").to(device)
    if os.path.exists(args.gen_path):
        print(f"[INFO] Loading trained generator weights from: {args.gen_path}")
        generator.load_state_dict(torch.load(args.gen_path, map_location=device))
    else:
        print(f"[WARNING] Generator checkpoint '{args.gen_path}' not found. Using initialized generator.")

    # Run Benchmark
    run_benchmark(
        model=model,
        generator=generator,
        test_loader=test_loader,
        device=device,
        save_dir=args.save_dir,
        num_visual_samples=args.num_visual_samples
    )


if __name__ == "__main__":
    main()
