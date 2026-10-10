"""
MedAdvR Testing & Robustness Benchmark Suite
============================================
Evaluates a trained segmentation model under clean and adversarial conditions.
Produces side-by-side diagnostic figures and clinical robustness scorecards.
"""

import os
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np

from dataset.loader import DatasetLoader
from dataset.train_test_split import get_train_test_loaders
from main_model.model import init_model
from generator.model import init_generator
from main_model.eval import compute_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="MedAdvR Testing & Benchmark Suite")
    parser.add_argument("--dataset_path", type=str, default="./data/glioma", help="Path to dataset")
    parser.add_argument("--dataset_type", type=str, default="tumor", help="Dataset type: tumor, extracted, dicom, nifti")
    parser.add_argument("--model_type", type=str, default="Unet", help="Model architecture")
    parser.add_argument("--model_path", type=str, default="outputs/model_final.pth", help="Path to trained model weights (optional)")
    parser.add_argument("--gen_path", type=str, default="outputs/generator_final.pth", help="Path to trained generator weights (optional)")
    parser.add_argument("--gen_type", type=str, default="edge", help="Attack type")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cuda, mps, cpu")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--save_dir", type=str, default="outputs/test_results")
    parser.add_argument("--num_visual_samples", type=int, default=5, help="Number of visual sample figures to export")
    return parser.parse_args()


def run_benchmark(model, generator, test_loader, device, save_dir, num_visual_samples=5):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    if generator:
        generator.eval()

    clean_metrics = {"dice": 0, "iou": 0, "precision": 0, "recall": 0}
    adv_metrics = {"dice": 0, "iou": 0, "precision": 0, "recall": 0}
    total_batches = 0

    visual_saved = 0

    with torch.no_grad():
        for batch_idx, (imgs, masks) in enumerate(test_loader):
            imgs, masks = imgs.to(device), masks.to(device)

            # 1. Clean Inference
            clean_logits = model(imgs)
            clean_preds = (torch.sigmoid(clean_logits) > 0.5).float()
            c_metrics = compute_metrics(clean_preds, masks)
            for k in clean_metrics:
                clean_metrics[k] += c_metrics[k]

            # 2. Adversarial Inference
            if generator:
                perturb = generator(imgs)
                adv_imgs = torch.clamp(imgs + perturb, 0.0, 1.0)
            else:
                # Fallback: random noise perturbation
                adv_imgs = torch.clamp(imgs + torch.randn_like(imgs) * 0.05, 0.0, 1.0)

            adv_logits = model(adv_imgs)
            adv_preds = (torch.sigmoid(adv_logits) > 0.5).float()
            a_metrics = compute_metrics(adv_preds, masks)
            for k in adv_metrics:
                adv_metrics[k] += a_metrics[k]

            total_batches += 1

            # 3. Save Visual Comparison Figures
            if visual_saved < num_visual_samples:
                for i in range(min(imgs.size(0), num_visual_samples - visual_saved)):
                    _save_diagnostic_plot(
                        img=imgs[i].cpu(),
                        mask=masks[i][0].cpu(),
                        clean_pred=clean_preds[i][0].cpu(),
                        adv_img=adv_imgs[i].cpu(),
                        adv_pred=adv_preds[i][0].cpu(),
                        sample_idx=visual_saved,
                        save_dir=save_dir
                    )
                    visual_saved += 1

    # Compute Averages
    for k in clean_metrics:
        clean_metrics[k] /= total_batches
        adv_metrics[k] /= total_batches

    robustness_retention = (adv_metrics["dice"] / (clean_metrics["dice"] + 1e-6)) * 100.0

    # Print Report
    print("\n" + "=" * 65)
    print("📊 MEDADVR COMPREHENSIVE TEST & ROBUSTNESS REPORT")
    print("=" * 65)
    print(f"Total Test Batches Evaluated: {total_batches}")
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
        f.write(f"Clean Dice Score      : {clean_metrics['dice']:.4f}\n")
        f.write(f"Attacked Dice Score   : {adv_metrics['dice']:.4f}\n")
        f.write(f"Clean IoU (Jaccard)   : {clean_metrics['iou']:.4f}\n")
        f.write(f"Attacked IoU (Jaccard): {adv_metrics['iou']:.4f}\n")
        f.write(f"Robustness Retention  : {robustness_retention:.2f}%\n")


def _save_diagnostic_plot(img, mask, clean_pred, adv_img, adv_pred, sample_idx, save_dir):
    """
    Saves a 5-panel diagnostic figure comparing Clean vs Perturbed conditions.
    """
    img_np = img.permute(1, 2, 0).numpy()
    adv_np = adv_img.permute(1, 2, 0).numpy()
    delta_np = np.abs(adv_np - img_np) * 5.0  # Magnified 5x for visibility

    plt.figure(figsize=(18, 4))

    # 1. Clean MRI Scan
    plt.subplot(1, 5, 1)
    plt.title("1. Clean Scan", fontsize=11, fontweight='bold')
    plt.imshow(img_np)
    plt.axis("off")

    # 2. Ground Truth Mask
    plt.subplot(1, 5, 2)
    plt.title("2. Ground Truth Mask", fontsize=11, fontweight='bold')
    plt.imshow(mask, cmap="gray")
    plt.axis("off")

    # 3. Clean Model Prediction
    plt.subplot(1, 5, 3)
    plt.title("3. Clean Prediction", fontsize=11, fontweight='bold')
    plt.imshow(clean_pred, cmap="Blues")
    plt.axis("off")

    # 4. Adversarial Attack (Magnified Delta)
    plt.subplot(1, 5, 4)
    plt.title("4. Attack Perturbation (5x)", fontsize=11, fontweight='bold')
    plt.imshow(np.clip(delta_np, 0, 1))
    plt.axis("off")

    # 5. Model Prediction under Attack
    plt.subplot(1, 5, 5)
    plt.title("5. Attacked Prediction", fontsize=11, fontweight='bold')
    plt.imshow(adv_pred, cmap="Reds")
    plt.axis("off")

    plt.tight_layout()
    plot_path = os.path.join(save_dir, f"diagnostic_sample_{sample_idx:03d}.png")
    plt.savefig(plot_path, dpi=150)
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

    print(f"[INFO] Running MedAdvR Testing on Device: {device}")

    # Dataset Loader
    dataset = DatasetLoader(args.dataset_path, dataset_type=args.dataset_type, img_size=args.img_size, batch_size=args.batch_size)
    _, test_loader = get_train_test_loaders(dataset, split_ratio=0.8, batch_size=args.batch_size)

    # Initialize Model & Generator
    model = init_model(args.model_type).to(device)
    if os.path.exists(args.model_path):
        print(f"[INFO] Loading trained model weights from: {args.model_path}")
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    else:
        print(f"[WARNING] Model checkpoint '{args.model_path}' not found. Using initialized weights for benchmark.")

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
