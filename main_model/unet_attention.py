import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        # Assuming g1 and x1 have same spatial dimensions. If not, g1 needs to be upsampled.
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=True)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class AttentionUNet(nn.Module):
    def __init__(self):
        super().__init__()

        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        self.enc1 = block(3, 64)
        self.enc2 = block(64, 128)
        self.enc3 = block(128, 256)
        self.pool = nn.MaxPool2d(2, 2)

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.ag3 = AttentionGate(F_g=256, F_l=128, F_int=64)
        self.dec3 = block(256 + 128, 128)
        
        self.ag2 = AttentionGate(F_g=128, F_l=64, F_int=32)
        self.dec2 = block(128 + 64, 64)

        self.out = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)               # [B, 64, H, W]
        x2 = self.enc2(self.pool(x1))   # [B, 128, H/2, W/2]
        x3 = self.enc3(self.pool(x2))   # [B, 256, H/4, W/4]

        # Decoder 1
        d2 = self.up(x3)                # [B, 256, H/2, W/2]
        x2_att = self.ag3(g=d2, x=x2)
        d2 = torch.cat([d2, x2_att], dim=1)
        d2 = self.dec3(d2)              # [B, 128, H/2, W/2]
        
        # Decoder 2
        d1 = self.up(d2)                # [B, 128, H, W]
        x1_att = self.ag2(g=d1, x=x1)
        d1 = torch.cat([d1, x1_att], dim=1)
        d1 = self.dec2(d1)              # [B, 64, H, W]

        return self.out(d1)
