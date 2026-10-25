"""
Multi-Task & Semantic Disconnect Min-Max Framework Orchestrator
===============================================================
Coordinates:
  1. Stratified 3-Way Dataset Split (Train 70% / Val 15% / Test 15%) across
     Glioma, Meningioma, and Pituitary tumors.
  2. Multi-Task Clean Pretraining (Segmentation + 3-Class Classification + Consistency Gate).
  3. Alternating Min-Max Cycles:
     - Phase A: Train Semantic Disconnect Generator (frozen defender)
     - Sample Export: Saves adversarial PNGs + Annotated Multi-Task Sample Cards + CSV Manifest
     - Phase B: Balanced 50/50 Clean + Adversarial Retraining (frozen generator)
     - Validation Checkpointing: Saves model_multitask_best.pth & generator_multitask_best.pth
  4. Final Unbiased Evaluation & 6-Panel Clinical Diagnostic Report on Reserved Test Set.
"""

import os
import torch

from dataset.multitask_loader import get_multitask_train_val_test_loaders
from main_model.multitask_unet import MultiTaskUNet
from generator.model import init_generator
from main_model.multitask_train import (
    train_multitask_clean,
    train_multitask_adv,
    evaluate_multitask_epoch,
)
from generator.multitask_train import train_generator_multitask
from utils.multitask_save import save_multitask_adv_samples
from test_multitask import run_multitask_benchmark


def FrameworkMultiTaskRun(
    dataset_path="./data/all_tumors",
    gen_type="edge",
    attack_mode="joint_disconnect",
    device="cpu",
    batch_size=16,
    img_size=128,
    lr_model=1e-3,
    lr_gen=1e-3,
    pretrain_epochs=5,
    cycles=3,
    gen_epochs=3,
    model_epochs=3,
    save_images=True,
    save_dir="outputs_multitask",
    max_buffer_size=5000,
    tau_audit=0.45,
    num_visual_samples=12,
):
    device = torch.device(device)
    os.makedirs(save_dir, exist_ok=True)
    print(f"[INFO] Initializing MedAdvR Phase 3 (Multi-Task Semantic Disconnect) on: {device}")

    # 1. Load Stratified 3-Way Multi-Task Split (70% Train / 15% Val / 15% Test)
    train_loader, val_loader, test_loader, base_ds = get_multitask_train_val_test_loaders(
        dataset_path=dataset_path,
        img_size=img_size,
        batch_size=batch_size,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
        augment_train=True,
    )
    print(f"[INFO] Dataset Loaded ({len(base_ds)} slices) | Class Breakdown: {base_ds.get_class_counts()}")
    print(f"[INFO] Splits -> Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")

    # 2. Initialize Dual-Head MultiTaskUNet & Generator
    model = MultiTaskUNet(num_classes=3, latent_dim=128).to(device)
    generator = init_generator("Unet").to(device)

    adv_buffer = []

    # 3. Clean Multi-Task Pretraining
    print("\n[INFO] Step 1: Pretraining Dual-Head MultiTaskUNet on Clean Data...")
    train_multitask_clean(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=pretrain_epochs,
        lr=lr_model,
    )

    best_composite_score = 0.0

    # 4. Alternating Min-Max Semantic Disconnect Cycles
    for cycle in range(cycles):
        print(f"\n==================== MULTI-TASK CYCLE {cycle} ====================")

        # ---------------------------------------------------------
        # Phase A: Train Semantic Disconnect Generator
        # ---------------------------------------------------------
        print(f"[INFO] Phase A: Training Generator (mode={attack_mode}, defender frozen)...")
        for p in model.parameters():
            p.requires_grad = False
        for p in generator.parameters():
            p.requires_grad = True

        new_adv_samples = train_generator_multitask(
            model=model,
            generator=generator,
            dataloader=train_loader,
            device=device,
            epochs=gen_epochs,
            lr=lr_gen,
            gen_type=gen_type,
            attack_mode=attack_mode,
        )

        # Update replay buffer
        adv_buffer.extend(new_adv_samples)
        if len(adv_buffer) > max_buffer_size:
            adv_buffer = adv_buffer[-max_buffer_size:]

        # Save Adversarial Samples + Annotated Multi-Task Cards (with Class Predictions!)
        if save_images:
            cycle_dir = os.path.join(save_dir, "adv_samples", f"cycle_{cycle}")
            save_multitask_adv_samples(
                adv_samples=new_adv_samples,
                save_dir=cycle_dir,
                model=model,
                device=device,
                max_cards=10,
                tau_audit=tau_audit,
            )
            print(f"[INFO] Saved Cycle {cycle} samples & annotated class prediction cards to: {cycle_dir}")

        # ---------------------------------------------------------
        # Phase B: Harden Multi-Task Defender & Consistency Gate
        # ---------------------------------------------------------
        print("[INFO] Phase B: Hardening Defender & Consistency Gate on 50/50 Balanced Batches...")
        for p in model.parameters():
            p.requires_grad = True
        for p in generator.parameters():
            p.requires_grad = False

        train_multitask_adv(
            model=model,
            train_loader=train_loader,
            adv_buffer=adv_buffer,
            device=device,
            epochs=model_epochs,
            lr=lr_model,
            batch_size=batch_size,
        )

        # Validation Checkpointing
        val_stats = evaluate_multitask_epoch(model, val_loader, device)
        composite = 0.5 * val_stats["dice"] + 0.5 * val_stats["cls_acc"]
        print(
            f"[CYCLE {cycle} VAL] Val Dice: {val_stats['dice']:.4f} | "
            f"Val Cls Acc: {val_stats['cls_acc']*100:.2f}% | "
            f"Composite Score: {composite:.4f}"
        )

        if composite > best_composite_score:
            best_composite_score = composite
            torch.save(model.state_dict(), os.path.join(save_dir, "model_multitask_best.pth"))
            torch.save(generator.state_dict(), os.path.join(save_dir, "generator_multitask_best.pth"))
            print(f"🔥 [CHECKPOINT] Saved new best Multi-Task model (Composite: {best_composite_score:.4f})")

    # 5. Save Final Checkpoints
    final_model_path = os.path.join(save_dir, "model_multitask_final.pth")
    final_gen_path = os.path.join(save_dir, "generator_multitask_final.pth")
    torch.save(model.state_dict(), final_model_path)
    torch.save(generator.state_dict(), final_gen_path)
    if not os.path.exists(os.path.join(save_dir, "model_multitask_best.pth")):
        torch.save(model.state_dict(), os.path.join(save_dir, "model_multitask_best.pth"))
        torch.save(generator.state_dict(), os.path.join(save_dir, "generator_multitask_best.pth"))

    print(f"\n[INFO] Saved final checkpoints to: {save_dir}/")

    # 6. Run Final Unseen Test Benchmark & Export 6-Panel Clinical Diagnostics
    print("\n[INFO] Running Unbiased Multi-Task Evaluation on Reserved Test Set...")
    run_multitask_benchmark(
        model=model,
        generator=generator,
        test_loader=test_loader,
        device=device,
        save_dir=os.path.join(save_dir, "test_results"),
        num_visual_samples=num_visual_samples,
        tau_audit=tau_audit,
    )

    return model, generator
