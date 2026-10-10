import torch
import os

from dataset.loader import DatasetLoader
from dataset.train_test_split import get_train_val_test_loaders
from main_model.model import init_model
from generator.model import init_generator
from main_model.train import train_model_clean, train_model_adv
from generator.train import train_generator
from main_model.eval import clean_evaluations
from utils.save import save_adv_samples


def FrameworkRun(
    dataset_path,
    dataset_type='tumor',
    model_type='Unet',
    gen_type='edge',
    device='cpu',
    batch_size=16,
    img_size=128,
    lr_model=1e-3,
    lr_gen=1e-3,
    pretrain_epochs=5,
    cycles=5,
    gen_epochs=3,
    model_epochs=3,
    save_images=True,
    save_dir="outputs",
    max_buffer_size=5000
):
    device = torch.device(device)
    print(f"[INFO] Initializing MedAdvR on device: {device}")

    # 🔴 1. Load dataset with 3-way split (Train 70% / Val 15% / Test 15%)
    dataset = DatasetLoader(
        dataset_path,
        dataset_type=dataset_type,
        img_size=img_size,
        batch_size=batch_size,
        augment=True
    )

    train_loader, val_loader, test_loader = get_train_val_test_loaders(
        dataset=dataset,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        batch_size=batch_size,
        seed=42
    )

    # 🔴 2. Initialize model and generator
    model = init_model(model_type).to(device)
    generator = init_generator("Unet").to(device)

    # 🔴 3. Adversarial buffer
    adv_buffer = []

    # 🔴 4. Pretrain model on clean data (stepped by val_loader to prevent leakage)
    print("\n[INFO] Pretraining model on clean data...")
    train_model_clean(
        model=model,
        train=train_loader,
        val=val_loader,
        device=device,
        epochs=pretrain_epochs,
        lr=lr_model
    )

    # 🔴 5. Min-Max Cycles
    for cycle in range(cycles):
        print(f"\n========== CYCLE {cycle} ==========")

        # ============================
        # 🔵 Phase A: Train Generator
        # ============================
        print("[INFO] Training Generator (model frozen)...")
        for param in model.parameters():
            param.requires_grad = False
        for param in generator.parameters():
            param.requires_grad = True

        new_adv_samples = train_generator(
            model=model,
            generator=generator,
            dataset=dataset,
            device=device,
            epochs=gen_epochs,
            lr=lr_gen,
            gen_type=gen_type
        )

        # Store adversarial samples
        adv_buffer.extend(new_adv_samples)
        if len(adv_buffer) > max_buffer_size:
            adv_buffer = adv_buffer[-max_buffer_size:]

        # Optional: Save samples to disk
        if save_images:
            cycle_dir = os.path.join(save_dir, "adv_samples", f"cycle_{cycle}")
            save_adv_samples(new_adv_samples, cycle_dir)

        # ============================
        # 🔴 Phase B: Train Model (Balanced 50/50 Clean vs Adv)
        # ============================
        print("[INFO] Hardening Defender on balanced clean + adversarial batches...")
        for param in model.parameters():
            param.requires_grad = True
        for param in generator.parameters():
            param.requires_grad = False

        train_model_adv(
            model=model,
            dataset=dataset,
            adv_buffer=adv_buffer,
            device=device,
            epochs=model_epochs,
            lr=lr_model,
            batch_size=batch_size
        )

    # 🔴 6. Final Unbiased Benchmark Evaluation on Test Set
    print("\n[INFO] Running Final Evaluation on Reserved Test Set...")
    final_dice = clean_evaluations(model, test_loader, device, save_dir=os.path.join(save_dir, "test_eval"))
    print(f"\n[FINAL BENCHMARK] Unseen Test Set Dice: {final_dice:.4f}")

    # 🔴 7. Save Checkpoints
    os.makedirs(save_dir, exist_ok=True)
    model_save_path = os.path.join(save_dir, "model_final.pth")
    gen_save_path = os.path.join(save_dir, "generator_final.pth")
    torch.save(model.state_dict(), model_save_path)
    torch.save(generator.state_dict(), gen_save_path)
    print(f"[INFO] Saved model checkpoint to: {model_save_path}")
    print(f"[INFO] Saved generator checkpoint to: {gen_save_path}")

    print("\n[INFO] Training Complete.")
    return model, generator
