"""
Dual-Head Multi-Task UNet with Cross-Attention Consistency Gate
===============================================================
Simultaneously performs:
  1. Tumor Boundary Segmentation: seg_logits [B, 1, H, W]
  2. Pathology Classification:    cls_logits [B, 3] (Glioma, Meningioma, Pituitary)
  3. Cross-Task Consistency Gate: consistency_score [B, 1] in [0, 1]

Mathematical Foundation:
  - F_seg: Mask-conditioned foreground representation extracted via differentiable
           sigmoid attention pooling over decoder features + spatial morphology statistics.
  - F_cls: Global semantic representation extracted from encoder bottleneck.
  - Cross-Attention Consistency Gate measures congruence between (F_seg, mask morphology)
    and (F_cls, predicted class distribution). If Disconnect = (1 - Consistency) > tau_audit,
    the scan triggers a Clinical Safety Triage Flag.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossTaskConsistencyGate(nn.Module):
    """
    Bidirectional Cross-Attention Consistency Module between Segmentation and Classification.
    Detects 'Semantic Disconnect' attacks (e.g., Silent Tumor or Pathology Swapping).
    """

    def __init__(self, latent_dim=128, num_classes=3):
        super().__init__()
        self.q_proj = nn.Linear(latent_dim, 32)
        self.k_proj = nn.Linear(latent_dim, 32)
        self.v_proj = nn.Linear(latent_dim, 16)
        self.diff_proj = nn.Linear(latent_dim, 16)
        self.scale = 32 ** -0.5

        # Compact, well-balanced 40-dim audit vector:
        #   16 (cross-attended v) + 16 (projected latent diff) + 2 (cos_sim, l2_dist)
        #   + 3 (mask_area, mask_peak, mask_uncertainty) + num_classes (cls_probs)
        in_dim = 16 + 16 + 2 + 3 + num_classes
        self.audit_net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def forward(self, seg_latent, cls_latent, seg_logits, cls_logits):
        """
        Args:
            seg_latent: [B, D] mask-conditioned representation
            cls_latent: [B, D] pathology classification representation
            seg_logits: [B, 1, H, W] raw segmentation logits
            cls_logits: [B, K] raw classification logits
        Returns:
            consistency_score: [B, 1] in [0, 1] (1 = congruent, 0 = semantic disconnect)
        """
        seg_norm = F.normalize(seg_latent, p=2, dim=1)
        cls_norm = F.normalize(cls_latent, p=2, dim=1)

        # 1. Cross-attention: Classification queries Segmentation features
        q = self.q_proj(cls_norm).unsqueeze(1)  # [B, 1, 32]
        k = self.k_proj(seg_norm).unsqueeze(1)  # [B, 1, 32]
        v = self.v_proj(seg_norm).unsqueeze(1)  # [B, 1, 16]

        attn_gate = torch.sigmoid(torch.bmm(q, k.transpose(1, 2)) * self.scale)  # [B, 1, 1]
        fused = (attn_gate * v).squeeze(1)  # [B, 16]

        # 2. Normalized latent discrepancy features
        feat_diff = self.diff_proj(torch.abs(seg_norm - cls_norm))  # [B, 16]
        cos_sim = (seg_norm * cls_norm).sum(dim=1, keepdim=True)    # [B, 1] in [-1, 1]
        l2_dist = torch.norm(seg_norm - cls_norm, p=2, dim=1, keepdim=True)  # [B, 1]

        # 3. Differentiable mask morphology & confidence statistics
        seg_probs = torch.sigmoid(seg_logits)  # [B, 1, H, W]
        mask_area = seg_probs.mean(dim=(2, 3))  # [B, 1]
        mask_peak = seg_probs.amax(dim=(2, 3))  # [B, 1]
        mask_uncertainty = (
            -seg_probs * torch.log(seg_probs + 1e-6)
            - (1.0 - seg_probs) * torch.log(1.0 - seg_probs + 1e-6)
        ).mean(dim=(2, 3))

        cls_probs = F.softmax(cls_logits, dim=1)  # [B, K]

        audit_input = torch.cat([
            fused,
            feat_diff,
            cos_sim,
            l2_dist,
            mask_area,
            mask_peak,
            mask_uncertainty,
            cls_probs
        ], dim=1)

        # Combine learned audit logit with explicit cosine alignment prior
        audit_logit = self.audit_net(audit_input) + 1.5 * cos_sim
        consistency_score = torch.sigmoid(audit_logit)  # [B, 1]
        return consistency_score


class MultiTaskUNet(nn.Module):
    """
    Dual-Head Multi-Task Defender Architecture:
      - Shared 4-stage UNet Encoder
      - Skip-connected Segmentation Decoder + Mask-Conditioned Latent Projector
      - Bottleneck Multi-Class Pathology Classifier
      - Cross-Attention Consistency Gate for Semantic Disconnect Detection
    """

    def __init__(self, num_classes=3, latent_dim=128):
        super().__init__()
        self.num_classes = num_classes
        self.latent_dim = latent_dim

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )

        # =========================
        # 1. SHARED ENCODER
        # =========================
        self.enc1 = conv_block(3, 64)       # [B, 64, H, W]
        self.enc2 = conv_block(64, 128)     # [B, 128, H/2, W/2]
        self.enc3 = conv_block(128, 256)    # [B, 256, H/4, W/4]
        self.enc4 = conv_block(256, 512)    # [B, 512, H/8, W/8] (Bottleneck)

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        # =========================
        # 2. SEGMENTATION DECODER
        # =========================
        self.dec3 = conv_block(512 + 256, 256)
        self.dec2 = conv_block(256 + 128, 128)
        self.dec1 = conv_block(128 + 64, 64)
        self.seg_out = nn.Conv2d(64, 1, kernel_size=1)

        # Mask-Conditioned Segmentation Latent Projector:
        # Combines mask-weighted foreground features (64) + background context (64) + spatial centroid/area (4) -> latent_dim
        self.seg_latent_proj = nn.Sequential(
            nn.Linear(64 + 64 + 4, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, latent_dim)
        )

        # =========================
        # 3. CLASSIFICATION HEAD
        # =========================
        # Uses both Average and Max pooled bottleneck features (512 * 2 = 1024) + mid-level pooled features (256)
        self.cls_latent_proj = nn.Sequential(
            nn.Linear(512 * 2 + 256, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(inplace=True),
        )
        self.cls_out = nn.Linear(latent_dim, num_classes)

        # =========================
        # 4. CONSISTENCY GATE
        # =========================
        self.consistency_gate = CrossTaskConsistencyGate(
            latent_dim=latent_dim, num_classes=num_classes
        )

    def _extract_seg_latent(self, dec_feat, seg_logits):
        """
        Extracts a differentiable mask-conditioned latent vector F_seg [B, latent_dim]
        so that if an attack erases or shifts the segmentation mask, F_seg changes immediately.
        """
        B, C, H, W = dec_feat.shape
        seg_probs = torch.sigmoid(seg_logits)  # [B, 1, H, W]

        # Soft foreground pooling (features inside the predicted tumor mask)
        fg_sum = seg_probs.sum(dim=(2, 3), keepdim=False) + 1e-5  # [B, 1]
        fg_feat = (dec_feat * seg_probs).sum(dim=(2, 3)) / fg_sum  # [B, 64]

        # Global background context pooling
        bg_probs = 1.0 - seg_probs
        bg_sum = bg_probs.sum(dim=(2, 3), keepdim=False) + 1e-5
        bg_feat = (dec_feat * bg_probs).sum(dim=(2, 3)) / bg_sum  # [B, 64]

        # Differentiable normalized spatial centroid (y_center, x_center) & area stats
        grid_y = torch.linspace(0, 1, H, device=dec_feat.device).view(1, 1, H, 1)
        grid_x = torch.linspace(0, 1, W, device=dec_feat.device).view(1, 1, 1, W)
        cy = (seg_probs * grid_y).sum(dim=(2, 3)) / fg_sum  # [B, 1]
        cx = (seg_probs * grid_x).sum(dim=(2, 3)) / fg_sum  # [B, 1]
        area_ratio = seg_probs.mean(dim=(2, 3))             # [B, 1]
        max_conf = seg_probs.amax(dim=(2, 3))               # [B, 1]

        morph_stats = torch.cat([cy, cx, area_ratio, max_conf], dim=1)  # [B, 4]
        combined = torch.cat([fg_feat, bg_feat, morph_stats], dim=1)    # [B, 132]
        return self.seg_latent_proj(combined)                           # [B, latent_dim]

    def forward(self, x, return_latents=False):
        # 1. Shared Encoder
        x1 = self.enc1(x)                # [B, 64, H, W]
        x2 = self.enc2(self.pool(x1))    # [B, 128, H/2, W/2]
        x3 = self.enc3(self.pool(x2))    # [B, 256, H/4, W/4]
        x4 = self.enc4(self.pool(x3))    # [B, 512, H/8, W/8]

        # 2. Segmentation Decoder
        d3 = self.up(x4)
        if d3.shape[2:] != x3.shape[2:]:
            d3 = F.interpolate(d3, size=x3.shape[2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, x3], dim=1))

        d2 = self.up(d3)
        if d2.shape[2:] != x2.shape[2:]:
            d2 = F.interpolate(d2, size=x2.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, x2], dim=1))

        d1 = self.up(d2)
        if d1.shape[2:] != x1.shape[2:]:
            d1 = F.interpolate(d1, size=x1.shape[2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, x1], dim=1))

        seg_logits = self.seg_out(d1)    # [B, 1, H, W]

        # 3. Classification Head
        gap4 = F.adaptive_avg_pool2d(x4, 1).flatten(1)  # [B, 512]
        gmp4 = F.adaptive_max_pool2d(x4, 1).flatten(1)  # [B, 512]
        gap3 = F.adaptive_avg_pool2d(x3, 1).flatten(1)  # [B, 256]
        cls_Pool = torch.cat([gap4, gmp4, gap3], dim=1) # [B, 1280]

        cls_latent = self.cls_latent_proj(cls_Pool)     # [B, latent_dim]
        cls_logits = self.cls_out(cls_latent)           # [B, num_classes]

        # 4. Mask-Conditioned Latent & Consistency Gate
        seg_latent = self._extract_seg_latent(d1, seg_logits)  # [B, latent_dim]
        consistency_score = self.consistency_gate(
            seg_latent, cls_latent, seg_logits, cls_logits
        )  # [B, 1]

        if return_latents:
            return seg_logits, cls_logits, consistency_score, seg_latent, cls_latent
        return seg_logits, cls_logits, consistency_score
