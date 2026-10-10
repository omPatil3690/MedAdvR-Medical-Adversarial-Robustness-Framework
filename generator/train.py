import torch
import torch.nn as nn
import torch.optim as optim
import logging

from utils.metric import iou_loss, dice_loss, iou_score, dice_score
from utils.train_helper import _resolve_loader
from utils.gen_losses import specialized_loss, realism_score, realism_loss, edge_diff_score


# =====================================================================
# 🏋️ TRAIN GEN EPOCH (Weighted per-sample calculation)
# =====================================================================
def _train_gen_epoch(generator, model, dataloader, optimizer, device, gen_type, lambda_real, tau, epoch, epochs):
    generator.train()
    model.eval()

    total_loss = 0.0
    total_attack = 0.0
    total_special = 0.0
    total_real = 0.0
    total_realism_score = 0.0
    total_iou_score = 0.0
    total_samples = 0
    adv_samples = []

    for batch_idx, (x, y) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)
        if y.dim() == 3:
            y = y.unsqueeze(1)
        y = (y > 0.5).float()

        b_size = x.size(0)

        # Generate perturbation
        perturb = generator(x)
        x_adv = torch.clamp(x + perturb, 0.0, 1.0)

        # Forward through frozen model (gradients flow back to generator)
        pred = model(x_adv)

        # Losses
        # L_attack: maximize failure = minimize IoU
        # iou_loss = 1 - IoU, so we negate it to make minimizing loss = maximizing failure
        L_attack = -iou_loss(pred, y)
        L_special = specialized_loss(x_adv, x, gen_type)
        L_real = realism_loss(x, x_adv, gen_type, tau=tau)
        R = realism_score(x, x_adv, gen_type)

        # Total loss
        loss = L_attack + L_special + lambda_real * L_real

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate metrics (weighted by batch size)
        total_loss += loss.item() * b_size
        total_attack += L_attack.item() * b_size
        total_special += L_special.item() * b_size
        total_real += L_real.item() * b_size
        
        r_val = R.item() if isinstance(R, torch.Tensor) else float(R)
        total_realism_score += r_val * b_size

        # Compute IoU score for logging
        with torch.no_grad():
            preds_binary = (torch.sigmoid(pred) > 0.5).float()
            iou = iou_score(preds_binary, y).item()
            total_iou_score += iou * b_size

        # Store adversarial samples in standard [C, H, W] float format
        adv_samples.extend([(x_adv[i].detach().cpu(), y[i].detach().cpu()) for i in range(b_size)])
        total_samples += b_size

        # Per-batch logging
        if batch_idx % 5 == 0:
            logging.info(
                f"Epoch {epoch+1:02d}/{epochs} | "
                f"Batch {batch_idx:3d} | "
                f"Loss: {loss.item():.4f} | "
                f"Attack: {L_attack.item():.4f} | "
                f"Special: {L_special.item():.4f} | "
                f"R_score: {r_val:.4f} | "
                f"R_loss: {L_real.item():.4f} | "
                f"IoU: {iou:.4f} | "
                f"λ: {lambda_real:.4f}"
            )

    if total_samples == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, adv_samples

    return (
        total_loss / total_samples,
        total_attack / total_samples,
        total_special / total_samples,
        total_real / total_samples,
        total_realism_score / total_samples,
        total_iou_score / total_samples,
        adv_samples
    )


# =====================================================================
# 🚀 MAIN TRAIN GENERATOR
# =====================================================================
def train_generator(model, generator, dataset, device, epochs, lr, gen_type='edge', tau=0.9, lambda_init=0.1):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(message)s',
        handlers=[
            logging.FileHandler("generator_training.log"),
            logging.StreamHandler()
        ],
        force=True
    )

    generator = generator.to(device)
    model = model.to(device)

    optimizer = optim.Adam(generator.parameters(), lr=lr)
    lambda_real = lambda_init

    dataloader = _resolve_loader(dataset)

    logging.info(f"Starting generator training for {gen_type} type...")
    logging.info(f"Total epochs: {epochs} | Batches per epoch: {len(dataloader)}\n")

    all_adv_samples = []

    for epoch in range(epochs):
        loss, attack, special, real, realism_score_avg, iou, adv_samples = _train_gen_epoch(
            generator, model, dataloader, optimizer, device, gen_type, lambda_real, tau, epoch, epochs
        )

        # Adaptive lambda
        if real > 1e-6:
            lambda_real *= 1.1
        else:
            lambda_real *= 0.9
        lambda_real = max(0.01, min(lambda_real, 10.0))

        logging.info(
            f"\n>>> EPOCH {epoch+1:02d}/{epochs} SUMMARY <<<\n"
            f"  Avg Loss:        {loss:.4f}\n"
            f"  Avg Attack:      {attack:.4f}\n"
            f"  Avg Special:     {special:.4f}\n"
            f"  Avg R_score:     {realism_score_avg:.4f}  (target: {tau})\n"
            f"  Avg R_loss:      {real:.4f}\n"
            f"  Avg IoU:         {iou:.4f}\n"
            f"  Lambda:          {lambda_real:.4f}\n"
        )

        all_adv_samples.extend(adv_samples)

    logging.info("Generator training complete.\n")
    return all_adv_samples