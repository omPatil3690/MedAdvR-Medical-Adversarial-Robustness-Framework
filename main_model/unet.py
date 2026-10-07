import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Convolutional Block
        # WHY:
        # - Two conv layers increase receptive field and feature richness
        # - BatchNorm stabilizes training (important for medical images)
        # - ReLU adds non-linearity
        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),

                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )

        # =========================
        # ENCODER (Downsampling)
        # =========================

        # WHY:
        # - Increasing channels = capturing more abstract features
        # - Spatial size reduces → semantic understanding increases

        self.enc1 = block(3, 64)      # Input RGB → low-level features
        self.enc2 = block(64, 128)    # Mid-level features
        self.enc3 = block(128, 256)   # High-level semantic features

        # WHY MaxPool:
        # - Standard in UNet
        # - Preserves strong activations (edges, structures)
        self.pool = nn.MaxPool2d(2, 2)

        # =========================
        # ECODER (Upsampling)
        # =========================

        # WHY:
        # - Skip connections help recover spatial detail lost in pooling
        # - Concatenation gives both coarse + fine features

        self.dec3 = block(256 + 128, 128)  # combine x3 (upsampled) + x2
        self.dec2 = block(128 + 64, 64)    # combine previous + x1

        # =========================
        # 🎯 OUTPUT LAYER
        # =========================

        # WHY:
        # - 1 channel → binary segmentation mask
        # - NO sigmoid here (important)
        #   → because BCEWithLogitsLoss expects raw logits
        self.out = nn.Conv2d(64, 1, kernel_size=1)

        # =========================
        # UPSAMPLING
        # =========================

        # WHY bilinear instead of transpose conv:
        # - avoids checkerboard artifacts
        # - more stable for medical segmentation
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, x):

        # =========================
        # 🔻 ENCODER FORWARD
        # =========================

        x1 = self.enc1(x)               # [B, 64, H, W]
        x2 = self.enc2(self.pool(x1))   # [B, 128, H/2, W/2]
        x3 = self.enc3(self.pool(x2))   # [B, 256, H/4, W/4]

        # =========================
        # 🔺 DECODER FORWARD
        # =========================

        # Step 1: Upsample + skip connection
        x = self.up(x3)                 # [B, 256, H/2, W/2]

        # IMPORTANT:
        # If sizes mismatch due to rounding, fix it
        if x.shape != x2.shape:
            x = F.interpolate(x, size=x2.shape[2:], mode="bilinear", align_corners=False)

        x = torch.cat([x, x2], dim=1)   # combine encoder + decoder features
        x = self.dec3(x)

        # Step 2: Upsample again
        x = self.up(x)                  # [B, 128, H, W]

        if x.shape != x1.shape:
            x = F.interpolate(x, size=x1.shape[2:], mode="bilinear", align_corners=False)

        x = torch.cat([x, x1], dim=1)
        x = self.dec2(x)

        # =========================
        # 🎯 OUTPUT
        # =========================

        logits = self.out(x)

        # IMPORTANT:
        # DO NOT apply sigmoid here
        # → handled in loss or evaluation
        return logits