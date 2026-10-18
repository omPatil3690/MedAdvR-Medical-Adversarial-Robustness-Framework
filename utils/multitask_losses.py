"""
Multi-Task & Semantic Disconnect Loss Module
============================================
Implements:
  1. Vectorized GPU SSIM & Ortho-Normalized Frequency Realism Losses
  2. Multi-Task Defender Loss (Segmentation + Classification + Cross-Task Alignment + Gate Audit)
  3. Semantic Disconnect Attacker Loss:
       L_attack = α * L_seg_failure + β * L_cls_mislead + γ * D_KL(F_seg || F_cls)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional.image import structural_similarity_index_measure as ssim

from .metric import iou_loss, dice_loss


# =====================================================================
# 1. VECTORIZED REALISM & SPECIALIZED GENERATOR LOSSES (Bug-Fixed)
# =====================================================================
def compute_ssim_fast(x1, x2, data_range=1.0):
    """Vectorized batch SSIM on GPU (avoids per-sample Python loop overhead)."""
    return ssim(x1, x2, data_range=data_range)


def edge_loss_mt(x_adv, x):
    grad_adv = torch.gradient(x_adv, dim=[2, 3])
    grad_x = torch.gradient(x, dim=[2, 3])
    diff = torch.abs(grad_adv[0] - grad_x[0]) + torch.abs(grad_adv[1] - grad_x[1])
    return -torch.mean(diff)


def intensity_loss_mt(x_adv, x):
    return -torch.mean(torch.abs(x_adv - x))


def texture_loss_mt(x_adv, x):
    """Ortho-normalized 2D FFT loss (prevents H*W magnitude scaling explosion)."""
    f_x = torch.fft.fft2(x, norm="ortho")
    f_x_adv = torch.fft.fft2(x_adv, norm="ortho")
    return -torch.mean(torch.abs(f_x - f_x_adv))


def specialized_loss_mt(x_adv, x, gen_type="edge"):
    if gen_type == "edge":
        return edge_loss_mt(x_adv, x)
    elif gen_type == "intensity":
        return intensity_loss_mt(x_adv, x)
    elif gen_type == "texture":
        return texture_loss_mt(x_adv, x)
    return torch.tensor(0.0, device=x.device)


def realism_score_mt(x, x_adv, gen_type="edge"):
    ssim_val = compute_ssim_fast(x, x_adv)
    if gen_type == "edge":
        intensity_dev = torch.mean(torch.abs(x_adv - x))
        return 0.5 * ssim_val + 0.5 * (1.0 - intensity_dev)
    elif gen_type == "intensity":
        grad1 = torch.gradient(x, dim=[2, 3])
        grad2 = torch.gradient(x_adv, dim=[2, 3])
        edge_diff = torch.mean(torch.abs(grad1[0] - grad2[0]) + torch.abs(grad1[1] - grad2[1]))
        return 0.5 * ssim_val + 0.5 * (1.0 - edge_diff)
    elif gen_type == "texture":
        f_x = torch.fft.fft2(x, norm="ortho")
        f_x_adv = torch.fft.fft2(x_adv, norm="ortho")
        freq_dev = torch.clamp(torch.mean(torch.abs(f_x - f_x_adv)), 0.0, 1.0)
        return 0.5 * ssim_val + 0.5 * (1.0 - freq_dev)
    return ssim_val


def realism_loss_mt(x, x_adv, gen_type="edge", tau=0.9):
    R = realism_score_mt(x, x_adv, gen_type)
    return torch.relu(tau - R), R


# =====================================================================
# 2. MULTI-TASK DEFENDER LOSS (Joint Seg + Cls + Consistency Gate)
# =====================================================================
def compute_latent_kl_divergence(f_seg, f_cls, temperature=1.0):
    """
    Computes symmetric KL divergence D_KL(P_seg || P_cls) between normalized
    latent feature distributions F_seg and F_cls.
    """
    log_p_seg = F.log_softmax(f_seg / temperature, dim=1)
    p_seg = F.softmax(f_seg / temperature, dim=1)
    log_p_cls = F.log_softmax(f_cls / temperature, dim=1)
    p_cls = F.softmax(f_cls / temperature, dim=1)

    kl_1 = F.kl_div(log_p_seg, p_cls, reduction="batchmean")
    kl_2 = F.kl_div(log_p_cls, p_seg, reduction="batchmean")
    return 0.5 * (kl_1 + kl_2)


def multitask_defender_loss(
    model,
    seg_logits,
    cls_logits,
    consistency_score,
    f_seg,
    f_cls,
    masks,
    labels,
    is_clean_flag=None,
    pos_weight_val=2.5,
    w_seg=1.0,
    w_cls=0.6,
    w_align=0.25,
    w_gate=0.4
):
    """
    Computes the complete Multi-Task Defender Loss:
      1. L_seg:   0.5 * BCEWithLogits + 0.5 * IoU_Loss
      2. L_cls:   CrossEntropyLoss on tumor pathology (Glioma / Meningioma / Pituitary)
      3. L_align: Pulls F_seg and F_cls together on clean samples via Cosine + KL alignment
      4. L_gate:  Trains the Cross-Attention Consistency Gate to output 1.0 on congruent
                  clean scans and 0.0 on mismatched / adversarial semantic disconnect scans.
    """
    device = seg_logits.device
    B = seg_logits.size(0)

    # 1. Segmentation Loss
    pos_weight = torch.tensor([pos_weight_val], device=device)
    bce_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    l_seg = 0.5 * bce_fn(seg_logits, masks) + 0.5 * iou_loss(seg_logits, masks)

    # 2. Classification Loss
    l_cls = F.cross_entropy(cls_logits, labels)

    # 3. Cross-Task Feature Alignment (applied to clean samples or all if is_clean_flag is None)
    if is_clean_flag is None:
        cos_align = 1.0 - F.cosine_similarity(f_seg, f_cls, dim=1).mean()
        kl_align = compute_latent_kl_divergence(f_seg, f_cls)
        l_align = 0.5 * cos_align + 0.5 * kl_align

        # 4. Consistency Gate Training (Clean = 1.0 + Synthetic Mismatched Negative Pairs = 0.0)
        pos_targets = torch.ones_like(consistency_score)
        l_gate_pos = F.binary_cross_entropy(consistency_score, pos_targets)

        # Create synthetic semantic disconnect pairs (roll batch by 1 and suppress/invert mask logits)
        if B > 1:
            rolled_cls_latent = torch.roll(f_cls, shifts=1, dims=0)
            rolled_cls_logits = torch.roll(cls_logits, shifts=1, dims=0)
            fake_empty_seg_logits = seg_logits - 6.0  # Simulate silent tumor (suppressed mask)
            neg_score_1 = model.consistency_gate(f_seg.detach(), rolled_cls_latent.detach(), fake_empty_seg_logits.detach(), rolled_cls_logits.detach())
            neg_targets = torch.zeros_like(neg_score_1)
            l_gate_neg = F.binary_cross_entropy(neg_score_1, neg_targets)
            l_gate = 0.5 * (l_gate_pos + l_gate_neg)
        else:
            l_gate = l_gate_pos
    else:
        # Balanced Batch Mode: is_clean_flag is 1.0 for clean samples, 0.0 for adversarial disconnect samples
        clean_mask = (is_clean_flag > 0.5)
        if clean_mask.any():
            cos_align = 1.0 - F.cosine_similarity(f_seg[clean_mask], f_cls[clean_mask], dim=1).mean()
            kl_align = compute_latent_kl_divergence(f_seg[clean_mask], f_cls[clean_mask])
            l_align = 0.5 * cos_align + 0.5 * kl_align
        else:
            l_align = torch.tensor(0.0, device=device)

        gate_targets = is_clean_flag.view(-1, 1).float()
        l_gate = F.binary_cross_entropy(consistency_score, gate_targets)

    total_loss = (w_seg * l_seg) + (w_cls * l_cls) + (w_align * l_align) + (w_gate * l_gate)
    return total_loss, {
        "seg_loss": l_seg.item(),
        "cls_loss": l_cls.item(),
        "align_loss": l_align.item(),
        "gate_loss": l_gate.item(),
    }


# =====================================================================
# 3. SEMANTIC DISCONNECT ATTACK LOSS (Generator Optimization)
# =====================================================================
def semantic_disconnect_attack_loss(
    seg_logits_adv,
    cls_logits_adv,
    f_seg_adv,
    f_cls_adv,
    y_mask,
    y_cls,
    num_classes=3,
    attack_mode="joint_disconnect",
    alpha=1.0,
    beta=0.8,
    gamma=0.5
):
    """
    Mathematical Formulation:
      L_attack = α * L_seg_failure(f_seg(x_adv), y_seg)
               + β * L_cls_mislead(f_cls(x_adv), y_target)
               + γ * D_disconnect(F_seg || F_cls)

    Attack Modes:
      - 'silent_tumor':
          Suppresses segmentation mask to zero while forcing high confidence in original tumor class.
      - 'pathology_swap':
          Shifts classification to wrong tumor class (y_target = (y_cls + 1) % K) while degrading mask.
      - 'joint_disconnect' (Default):
          Combines mask destruction (-IoU_loss), targeted pathology swapping, and latent feature
          decoupling (minimizing cosine similarity & maximizing KL divergence between F_seg and F_cls).
    """
    target_swap_cls = (y_cls + 1) % num_classes

    if attack_mode == "silent_tumor":
        # Force segmentation mask toward empty (0) while keeping class prediction confident
        zero_mask = torch.zeros_like(y_mask)
        l_seg_fail = F.binary_cross_entropy_with_logits(seg_logits_adv, zero_mask)
        l_cls_mislead = F.cross_entropy(cls_logits_adv, y_cls)  # Maintain high confidence in tumor presence
    elif attack_mode == "pathology_swap":
        # Corrupt segmentation mask and force classifier into target swapped pathology
        l_seg_fail = -iou_loss(seg_logits_adv, y_mask)
        l_cls_mislead = F.cross_entropy(cls_logits_adv, target_swap_cls)
    else:
        # 'joint_disconnect': Full cross-task contradiction
        l_seg_fail = -iou_loss(seg_logits_adv, y_mask)
        l_cls_mislead = F.cross_entropy(cls_logits_adv, target_swap_cls)

    # Latent Feature Contradiction: push F_seg_adv and F_cls_adv apart
    # Minimizing (cos_sim - D_KL) forces orthogonal/opposing latents and high KL divergence
    cos_sim = F.cosine_similarity(f_seg_adv, f_cls_adv, dim=1).mean()
    kl_div = compute_latent_kl_divergence(f_seg_adv, f_cls_adv)
    l_latent_contradiction = cos_sim - torch.clamp(kl_div, max=2.0)

    l_attack = (alpha * l_seg_fail) + (beta * l_cls_mislead) + (gamma * l_latent_contradiction)
    return l_attack, {
        "seg_fail": l_seg_fail.item(),
        "cls_mislead": l_cls_mislead.item(),
        "latent_contradiction": l_latent_contradiction.item(),
    }
