"""
Multi-Task Semantic Disconnect Generator Trainer
================================================
Trains the adversarial generator to induce Cross-Task Semantic Contradictions:
  - Destroys segmentation mask overlap (-IoU_Loss)
  - Misleads pathology classifier into wrong tumor subtype (or Silent Tumor mode)
  - Decouples latent representations F_seg and F_cls
  - Preserves clinical realism via adaptive barrier penalty (R >= tau)

Bug Fix Included:
  - Collects adversarial samples ONLY during the final converged epoch of each cycle
    (preventing early un-converged epoch samples from polluting the replay buffer).
"""

import torch
import torch.optim as optim
import logging

from utils.metric import iou_score
from utils.multitask_losses import (
    semantic_disconnect_attack_loss,
    specialized_loss_mt,
    realism_loss_mt,
)


def _train_multitask_gen_epoch(
    generator,
    model,
    dataloader,
    optimizer,
    device,
    gen_type="edge",
    attack_mode="joint_disconnect",
    lambda_real=0.1,
    tau=0.9,
    epoch=0,
    epochs=3,
    collect_samples=False
):
    generator.train()
    model.eval()

    total_loss = 0.0
    total_attack = 0.0
    total_special = 0.0
    total_real_loss = 0.0
    total_real_score = 0.0
    total_iou = 0.0
    total_cls_acc = 0.0
    total_samples = 0
    adv_samples = []

    for batch_idx, (x, y_mask, y_cls) in enumerate(dataloader):
        x = x.to(device)
        y_mask = y_mask.to(device)
        y_cls = y_cls.to(device)
        if y_mask.dim() == 3:
            y_mask = y_mask.unsqueeze(1)
        y_mask = (y_mask > 0.5).float()

        b_size = x.size(0)

        # 1. Generate perturbation & clamp to valid image manifold [0, 1]
        perturb = generator(x)
        x_adv = torch.clamp(x + perturb, 0.0, 1.0)

        # 2. Forward through frozen MultiTaskUNet (gradients flow back to generator)
        seg_logits_adv, cls_logits_adv, cons_adv, f_seg_adv, f_cls_adv = model(
            x_adv, return_latents=True
        )

        # 3. Compute Semantic Disconnect Attack Loss + Domain Special Loss + Realism Barrier
        L_attack, atk_details = semantic_disconnect_attack_loss(
            seg_logits_adv=seg_logits_adv,
            cls_logits_adv=cls_logits_adv,
            f_seg_adv=f_seg_adv,
            f_cls_adv=f_cls_adv,
            y_mask=y_mask,
            y_cls=y_cls,
            attack_mode=attack_mode,
        )
        L_special = specialized_loss_mt(x_adv, x, gen_type=gen_type)
        L_real, R_score = realism_loss_mt(x, x_adv, gen_type=gen_type, tau=tau)

        loss = L_attack + L_special + (lambda_real * L_real)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 4. Accumulate weighted metrics
        total_loss += loss.item() * b_size
        total_attack += L_attack.item() * b_size
        total_special += L_special.item() * b_size
        total_real_loss += L_real.item() * b_size
        r_val = R_score.item() if isinstance(R_score, torch.Tensor) else float(R_score)
        total_real_score += r_val * b_size

        with torch.no_grad():
            preds_bin = (torch.sigmoid(seg_logits_adv) > 0.5).float()
            batch_iou = iou_score(preds_bin, y_mask).item()
            pred_cls = torch.argmax(cls_logits_adv, dim=1)
            batch_cls_acc = (pred_cls == y_cls).float().mean().item()

            total_iou += batch_iou * b_size
            total_cls_acc += batch_cls_acc * b_size

        # Only store final-epoch converged adversarial samples
        if collect_samples:
            for i in range(b_size):
                adv_samples.append((
                    x_adv[i].detach().cpu(),
                    y_mask[i].detach().cpu(),
                    y_cls[i].detach().cpu(),
                    x[i].detach().cpu(),
                ))

        total_samples += b_size

        if batch_idx % 10 == 0:
            logging.info(
                f"Gen Epoch {epoch+1:02d}/{epochs} | Batch {batch_idx:3d} | "
                f"Loss: {loss.item():.4f} | Atk: {L_attack.item():.4f} | "
                f"R_score: {r_val:.4f} | Adv IoU: {batch_iou:.4f} | "
                f"Adv Cls Acc: {batch_cls_acc*100:.1f}% | λ: {lambda_real:.4f}"
            )

    if total_samples == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, adv_samples

    return (
        total_loss / total_samples,
        total_attack / total_samples,
        total_special / total_samples,
        total_real_loss / total_samples,
        total_real_score / total_samples,
        total_iou / total_samples,
        total_cls_acc / total_samples,
        adv_samples,
    )


def train_generator_multitask(
    model,
    generator,
    dataloader,
    device="cpu",
    epochs=3,
    lr=1e-3,
    gen_type="edge",
    attack_mode="joint_disconnect",
    tau=0.9,
    lambda_init=0.1,
):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.FileHandler("generator_multitask_training.log"),
            logging.StreamHandler(),
        ],
        force=True,
    )

    generator = generator.to(device)
    model = model.to(device)

    optimizer = optim.Adam(generator.parameters(), lr=lr)
    lambda_real = lambda_init

    logging.info(
        f"Starting Multi-Task Semantic Disconnect Generator ({gen_type} | mode={attack_mode})..."
    )

    final_adv_samples = []

    for epoch in range(epochs):
        is_last_epoch = (epoch == epochs - 1)
        loss, atk, spec, r_loss, r_score, adv_iou, adv_cls_acc, adv_samples = _train_multitask_gen_epoch(
            generator=generator,
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            gen_type=gen_type,
            attack_mode=attack_mode,
            lambda_real=lambda_real,
            tau=tau,
            epoch=epoch,
            epochs=epochs,
            collect_samples=is_last_epoch,
        )

        # Adaptive barrier penalty schedule
        if r_loss > 1e-6:
            lambda_real *= 1.1
        else:
            lambda_real *= 0.9
        lambda_real = max(0.01, min(lambda_real, 10.0))

        logging.info(
            f"\n>>> GEN EPOCH {epoch+1:02d}/{epochs} SUMMARY <<<\n"
            f"  Avg Total Loss:      {loss:.4f}\n"
            f"  Avg Attack Loss:     {atk:.4f}\n"
            f"  Avg Realism Score:   {r_score:.4f} (Target >= {tau})\n"
            f"  Attacked Seg IoU:    {adv_iou:.4f}\n"
            f"  Attacked Cls Acc:    {adv_cls_acc*100:.2f}%\n"
            f"  Updated Lambda (λ):  {lambda_real:.4f}\n"
        )

        if is_last_epoch:
            final_adv_samples = adv_samples

    return final_adv_samples
