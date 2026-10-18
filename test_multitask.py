"""
MedAdvR Multi-Task & Semantic Disconnect Benchmark Suite
========================================================
Evaluates the Dual-Head MultiTaskUNet under Clean and Semantic Disconnect Attack conditions:
  1. Segmentation Metrics: Dice, IoU (Jaccard), Precision, Recall
  2. Pathology Classification Metrics: Overall Top-1 Accuracy + Per-Class Accuracy
     (Glioma, Meningioma, Pituitary)
  3. Cross-Task Consistency & Safety Triage Audit:
     - Clean vs. Attacked Semantic Disconnect Score
     - Safety Triage Detection Rate (% of attacks intercepted by Consistency Gate)
  4. Exports 6-Panel Clinical Diagnostic Figures including Class Probability Bars
     and Safety Triage Status Banners.
"""

import os
import csv
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from dataset.multitask_loader import get_multitask_train_val_test_loaders, CLASS_NAMES
from main_model.multitask_unet import MultiTaskUNet
from generator.model import init_generator
from main_model.eval import compute_metrics
from utils.metric import dice_score


def parse_args():
    parser = argparse.ArgumentParser(description="MedAdvR Multi-Task Semantic Disconnect Benchmark")
    parser.add_argument("--dataset_path", type=str, default="./data/all_tumors", help="Path to dataset")
    parser.add_argument("--model_path", type=str, default="outputs_multitask/model_multitask_best.pth")
    parser.add_argument("--gen_path", type=str, default="outputs_multitask/generator_multitask_best.pth")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--tau_audit", type=float, default=0.45, help="Semantic disconnect threshold for Safety Triage")
    parser.add_argument("--save_dir", type=str, default="outputs_multitask/test_results")
    parser.add_argument("--num_visual_samples", type=int, default=10)
    return parser.parse_args()


def run_multitask_benchmark(
    model,
    generator,
    test_loader,
    device,
    save_dir="outputs_multitask/test_results",
    num_visual_samples=10,
    tau_audit=0.45
):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    if generator is not None:
        generator.eval()

    clean_seg = {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}
    adv_seg = {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}

    clean_cls_correct = 0
    adv_cls_correct = 0
    class_totals = {0: 0, 1: 0, 2: 0}
    class_clean_correct = {0: 0, 1: 0, 2: 0}
    class_adv_correct = {0: 0, 1: 0, 2: 0}

    clean_disconnect_sum = 0.0
    adv_disconnect_sum = 0.0
    triage_flagged_adv = 0
    semantic_contradictions = 0
    contradictions_caught = 0

    total_samples = 0
    visual_saved = 0
    csv_rows = []

    with torch.no_grad():
        for imgs, masks, labels in test_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            if masks.dim() == 3:
                masks = masks.unsqueeze(1)
            masks = (masks > 0.5).float()

            b_size = imgs.size(0)

            # 1. Clean Multi-Task Inference
            c_seg_logits, c_cls_logits, c_cons = model(imgs)
            c_preds_mask = (torch.sigmoid(c_seg_logits) > 0.5).float()
            c_probs = F.softmax(c_cls_logits, dim=1)
            c_preds_cls = torch.argmax(c_probs, dim=1)

            c_m = compute_metrics(c_preds_mask, masks)
            for k in clean_seg:
                clean_seg[k] += c_m[k] * b_size

            clean_cls_correct += (c_preds_cls == labels).sum().item()
            c_disc = 1.0 - c_cons.squeeze(1)
            clean_disconnect_sum += c_disc.sum().item()

            # 2. Adversarial Semantic Disconnect Inference
            if generator is not None:
                perturb = generator(imgs)
                adv_imgs = torch.clamp(imgs + perturb, 0.0, 1.0)
            else:
                adv_imgs = torch.clamp(imgs + 0.05 * torch.randn_like(imgs), 0.0, 1.0)

            a_seg_logits, a_cls_logits, a_cons = model(adv_imgs)
            a_preds_mask = (torch.sigmoid(a_seg_logits) > 0.5).float()
            a_probs = F.softmax(a_cls_logits, dim=1)
            a_preds_cls = torch.argmax(a_probs, dim=1)

            a_m = compute_metrics(a_preds_mask, masks)
            for k in adv_seg:
                adv_seg[k] += a_m[k] * b_size

            adv_cls_correct += (a_preds_cls == labels).sum().item()
            a_disc = 1.0 - a_cons.squeeze(1)
            adv_disconnect_sum += a_disc.sum().item()

            # Per-sample statistics & Safety Triage audit
            for i in range(b_size):
                y_true = int(labels[i].item())
                y_c = int(c_preds_cls[i].item())
                y_a = int(a_preds_cls[i].item())

                class_totals[y_true] = class_totals.get(y_true, 0) + 1
                if y_c == y_true:
                    class_clean_correct[y_true] += 1
                if y_a == y_true:
                    class_adv_correct[y_true] += 1

                sample_c_dice = dice_score(c_preds_mask[i:i+1], masks[i:i+1]).item()
                sample_a_dice = dice_score(a_preds_mask[i:i+1], masks[i:i+1]).item()
                sample_c_disc = float(c_disc[i].item())
                sample_a_disc = float(a_disc[i].item())

                is_flagged = sample_a_disc > tau_audit
                if is_flagged:
                    triage_flagged_adv += 1

                # Semantic contradiction occurs if class swapped OR mask dropped > 15%
                is_contradiction = (y_a != y_true) or (sample_a_dice < sample_c_dice - 0.15)
                if is_contradiction:
                    semantic_contradictions += 1
                    if is_flagged:
                        contradictions_caught += 1

                c_conf = float(c_probs[i, y_c].item()) * 100.0
                a_conf = float(a_probs[i, y_a].item()) * 100.0

                csv_rows.append({
                    "sample_idx": total_samples + i,
                    "true_class": CLASS_NAMES[y_true],
                    "clean_pred_class": CLASS_NAMES[y_c],
                    "clean_conf_pct": f"{c_conf:.1f}",
                    "clean_dice": f"{sample_c_dice:.4f}",
                    "clean_disconnect": f"{sample_c_disc:.4f}",
                    "adv_pred_class": CLASS_NAMES[y_a],
                    "adv_conf_pct": f"{a_conf:.1f}",
                    "adv_dice": f"{sample_a_dice:.4f}",
                    "adv_disconnect": f"{sample_a_disc:.4f}",
                    "triage_status": "REFER_TO_RADIOLOGIST" if is_flagged else "SAFE_PASS",
                })

                if visual_saved < num_visual_samples:
                    _save_multitask_diagnostic_plot(
                        img=imgs[i].cpu(),
                        mask=masks[i, 0].cpu(),
                        clean_pred_mask=c_preds_mask[i, 0].cpu(),
                        adv_img=adv_imgs[i].cpu(),
                        adv_pred_mask=a_preds_mask[i, 0].cpu(),
                        true_cls_idx=y_true,
                        clean_probs=c_probs[i].cpu().numpy(),
                        adv_probs=a_probs[i].cpu().numpy(),
                        clean_dice=sample_c_dice,
                        adv_dice=sample_a_dice,
                        clean_disc=sample_c_disc,
                        adv_disc=sample_a_disc,
                        tau_audit=tau_audit,
                        sample_idx=visual_saved,
                        save_dir=save_dir,
                    )
                    visual_saved += 1

            total_samples += b_size

    # Normalize metrics
    for k in clean_seg:
        clean_seg[k] /= max(1, total_samples)
        adv_seg[k] /= max(1, total_samples)

    clean_cls_acc = (clean_cls_correct / max(1, total_samples)) * 100.0
    adv_cls_acc = (adv_cls_correct / max(1, total_samples)) * 100.0
    avg_c_disc = clean_disconnect_sum / max(1, total_samples)
    avg_a_disc = adv_disconnect_sum / max(1, total_samples)
    triage_rate = (triage_flagged_adv / max(1, total_samples)) * 100.0
    catch_rate = (contradictions_caught / max(1, semantic_contradictions)) * 100.0 if semantic_contradictions > 0 else 100.0
    seg_retention = (adv_seg["dice"] / (clean_seg["dice"] + 1e-6)) * 100.0

    # Print Comprehensive Multi-Task & Triage Report
    print("\n" + "=" * 78)
    print("🧠 MEDADVR PHASE 3: MULTI-TASK & SEMANTIC DISCONNECT BENCHMARK REPORT")
    print("=" * 78)
    print(f"Total Held-Out Test Slices Evaluated: {total_samples}")
    print("-" * 78)
    print(f"{'Task / Metric':<28} | {'Clean Performance':<20} | {'Under Semantic Attack':<22}")
    print("-" * 78)
    print(f"{'Seg Dice Score':<28} | {clean_seg['dice']:<20.4f} | {adv_seg['dice']:<22.4f}")
    print(f"{'Seg IoU (Jaccard)':<28} | {clean_seg['iou']:<20.4f} | {adv_seg['iou']:<22.4f}")
    print(f"{'Seg Precision':<28} | {clean_seg['precision']:<20.4f} | {adv_seg['precision']:<22.4f}")
    print(f"{'Seg Recall':<28} | {clean_seg['recall']:<20.4f} | {adv_seg['recall']:<22.4f}")
    print("-" * 78)
    print(f"{'Pathology Cls Accuracy':<28} | {clean_cls_acc:<19.2f}% | {adv_cls_acc:<21.2f}%")
    for c_idx, c_name in enumerate(CLASS_NAMES):
        n_c = max(1, class_totals.get(c_idx, 0))
        c_acc_i = (class_clean_correct.get(c_idx, 0) / n_c) * 100.0
        a_acc_i = (class_adv_correct.get(c_idx, 0) / n_c) * 100.0
        print(f"  ├─ {c_name + ' Acc':<23} | {c_acc_i:<19.2f}% | {a_acc_i:<21.2f}%  (n={class_totals.get(c_idx, 0)})")
    print("-" * 78)
    print(f"{'Semantic Disconnect Score':<28} | {avg_c_disc:<20.4f} | {avg_a_disc:<22.4f} (τ={tau_audit})")
    print(f"🛡️  Segmentation Robustness Retention : {seg_retention:.2f}%")
    print(f"🚨 Safety Triage Detection Rate      : {triage_rate:.2f}% ({triage_flagged_adv}/{total_samples} attacked scans flagged)")
    print(f"🎯 Contradiction Interception Rate   : {catch_rate:.2f}% ({contradictions_caught}/{semantic_contradictions} contradictions caught)")
    print("=" * 78)
    print(f"💾 6-Panel Diagnostic Plots & CSV saved to: {save_dir}/")

    # Save CSV manifest
    if csv_rows:
        csv_path = os.path.join(save_dir, "test_multitask_predictions.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)

    # Save text scorecard
    scorecard_path = os.path.join(save_dir, "multitask_scorecard.txt")
    with open(scorecard_path, "w") as f:
        f.write("===== MEDADVR MULTI-TASK & SEMANTIC DISCONNECT SCORECARD =====\n\n")
        f.write(f"Total Test Samples Evaluated       : {total_samples}\n\n")
        f.write(f"Clean Segmentation Dice            : {clean_seg['dice']:.4f}\n")
        f.write(f"Attacked Segmentation Dice         : {adv_seg['dice']:.4f}\n")
        f.write(f"Clean Segmentation IoU             : {clean_seg['iou']:.4f}\n")
        f.write(f"Attacked Segmentation IoU          : {adv_seg['iou']:.4f}\n\n")
        f.write(f"Clean Pathology Accuracy           : {clean_cls_acc:.2f}%\n")
        f.write(f"Attacked Pathology Accuracy        : {adv_cls_acc:.2f}%\n\n")
        f.write(f"Clean Semantic Disconnect Score    : {avg_c_disc:.4f}\n")
        f.write(f"Attacked Semantic Disconnect Score : {avg_a_disc:.4f}\n")
        f.write(f"Safety Triage Interception Rate    : {catch_rate:.2f}%\n")

    return {
        "clean_dice": clean_seg["dice"],
        "adv_dice": adv_seg["dice"],
        "clean_cls_acc": clean_cls_acc,
        "adv_cls_acc": adv_cls_acc,
        "catch_rate": catch_rate,
    }


def _save_multitask_diagnostic_plot(
    img,
    mask,
    clean_pred_mask,
    adv_img,
    adv_pred_mask,
    true_cls_idx,
    clean_probs,
    adv_probs,
    clean_dice,
    adv_dice,
    clean_disc,
    adv_disc,
    tau_audit,
    sample_idx,
    save_dir
):
    """
    Exports a 6-panel clinical diagnostic figure containing:
      1. Clean MRI Scan + True Class
      2. Ground Truth Mask + Pathology Label
      3. Clean Multi-Task Prediction (Cyan Mask Overlay + Pred Class & Confidence + Dice)
      4. 10x Perturbation Heatmap
      5. Attacked Multi-Task Prediction (Red Mask Overlay + Attacked Pred Class & Confidence + Dice)
      6. Clinical Triage & Class Probability Comparison Panel (Clean vs Attacked bars + Disconnect Threshold)
    """
    img_np = img.permute(1, 2, 0).numpy()
    adv_np = adv_img.permute(1, 2, 0).numpy()
    mask_np = mask.numpy()
    c_mask_np = clean_pred_mask.numpy()
    a_mask_np = adv_pred_mask.numpy()

    true_cls_name = CLASS_NAMES[true_cls_idx]
    c_pred_idx = int(np.argmax(clean_probs))
    a_pred_idx = int(np.argmax(adv_probs))
    c_pred_name = CLASS_NAMES[c_pred_idx]
    a_pred_name = CLASS_NAMES[a_pred_idx]
    c_conf = float(clean_probs[c_pred_idx]) * 100.0
    a_conf = float(adv_probs[a_pred_idx]) * 100.0

    delta_np = np.abs(adv_np - img_np).mean(axis=2) * 10.0

    # Clean Overlay (Cyan)
    gray_c = img_np.mean(axis=2)
    clean_overlay = np.stack([gray_c, gray_c, gray_c], axis=-1)
    clean_overlay[c_mask_np > 0.5] = [0.1, 0.9, 0.9]

    # Attacked Overlay (Red)
    gray_a = adv_np.mean(axis=2)
    adv_overlay = np.stack([gray_a, gray_a, gray_a], axis=-1)
    adv_overlay[a_mask_np > 0.5] = [1.0, 0.2, 0.2]

    fig, axes = plt.subplots(1, 6, figsize=(24, 4.4), gridspec_kw={"width_ratios": [1, 1, 1, 1, 1, 1.25]})

    # Panel 1: Clean Scan
    axes[0].set_title(f"1. Clean MRI Scan\nTrue Class: {true_cls_name}", fontsize=10.5, fontweight="bold")
    axes[0].imshow(img_np)
    axes[0].axis("off")

    # Panel 2: Ground Truth Mask
    axes[1].set_title(f"2. Ground Truth Mask\nPathology: {true_cls_name}", fontsize=10.5, fontweight="bold")
    axes[1].imshow(mask_np, cmap="gray", vmin=0, vmax=1)
    axes[1].axis("off")

    # Panel 3: Clean Multi-Task Prediction
    c_color = "teal" if c_pred_idx == true_cls_idx else "darkorange"
    axes[2].set_title(
        f"3. Clean Pred (Dice: {clean_dice:.2f})\nCls: {c_pred_name} ({c_conf:.1f}%)",
        fontsize=10.5, fontweight="bold", color=c_color
    )
    axes[2].imshow(clean_overlay)
    axes[2].axis("off")

    # Panel 4: Perturbation Heatmap
    axes[3].set_title("4. Perturbation (10x)\nSemantic Disconnect δ", fontsize=10.5, fontweight="bold")
    axes[3].imshow(delta_np, cmap="magma", vmin=0, vmax=0.5)
    axes[3].axis("off")

    # Panel 5: Attacked Multi-Task Prediction
    axes[4].set_title(
        f"5. Attacked Pred (Dice: {adv_dice:.2f})\nCls: {a_pred_name} ({a_conf:.1f}%)",
        fontsize=10.5, fontweight="bold", color="darkred"
    )
    axes[4].imshow(adv_overlay)
    axes[4].axis("off")

    # Panel 6: Class Probability & Safety Triage Audit Panel
    ax6 = axes[5]
    y_pos = np.arange(len(CLASS_NAMES))
    bar_h = 0.34
    ax6.barh(y_pos + bar_h / 2, clean_probs * 100.0, height=bar_h, color="#17becf", label="Clean Prob %")
    ax6.barh(y_pos - bar_h / 2, adv_probs * 100.0, height=bar_h, color="#e63946", label="Attacked Prob %")
    ax6.set_yticks(y_pos)
    ax6.set_yticklabels(CLASS_NAMES, fontsize=9.5, fontweight="bold")
    ax6.set_xlim(0, 108)
    ax6.set_xlabel("Confidence (%)", fontsize=9)
    ax6.legend(loc="lower right", fontsize=8)
    ax6.grid(axis="x", linestyle="--", alpha=0.4)

    triage_flagged = adv_disc > tau_audit
    badge_text = (
        f"[ALERT: REFER TO RADIOLOGIST]\nDisconnect: {adv_disc:.2f} > τ ({tau_audit})"
        if triage_flagged
        else f"[PASS: CONSISTENT]\nDisconnect: {adv_disc:.2f} ≤ τ ({tau_audit})"
    )
    badge_color = "crimson" if triage_flagged else "seagreen"
    ax6.set_title(badge_text, fontsize=9.5, fontweight="bold", color=badge_color)

    plt.tight_layout()
    out_file = os.path.join(save_dir, f"multitask_diagnostic_{sample_idx:03d}_{true_cls_name.lower()}.png")
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            args.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"
    device = torch.device(args.device)

    print(f"[INFO] Running Multi-Task Benchmark on Device: {device}")

    _, _, test_loader, _ = get_multitask_train_val_test_loaders(
        dataset_path=args.dataset_path,
        img_size=args.img_size,
        batch_size=args.batch_size,
        seed=42,
        augment_train=False,
    )

    model = MultiTaskUNet(num_classes=3, latent_dim=128).to(device)
    if os.path.exists(args.model_path):
        print(f"[INFO] Loading MultiTaskUNet checkpoint: {args.model_path}")
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    else:
        print(f"[WARNING] Model checkpoint '{args.model_path}' not found. Using initialized weights.")

    generator = init_generator("Unet").to(device)
    if os.path.exists(args.gen_path):
        print(f"[INFO] Loading Generator checkpoint: {args.gen_path}")
        generator.load_state_dict(torch.load(args.gen_path, map_location=device))
    else:
        print(f"[WARNING] Generator checkpoint '{args.gen_path}' not found. Using initialized generator.")

    run_multitask_benchmark(
        model=model,
        generator=generator,
        test_loader=test_loader,
        device=device,
        save_dir=args.save_dir,
        num_visual_samples=args.num_visual_samples,
        tau_audit=args.tau_audit,
    )


if __name__ == "__main__":
    main()
