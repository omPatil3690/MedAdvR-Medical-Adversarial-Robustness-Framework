import torch.nn as nn
import torch
class UNet(nn.Module):
    def __init__(self, eps=0.05):
        super().__init__()
        self.eps = eps
        def block(in_c, out_c):
            return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
            )
        
        # Encoder
        self.enc1 = block(3, 32)
        self.enc2 = block(32, 64)
        self.enc3 = block(64, 128)
        self.pool = nn.AvgPool2d(2)
        
        # Decoder
        self.dec2 = block(128 + 64, 64)
        self.dec1 = block(64 + 32, 32)
        self.out = nn.Conv2d(32, 3, kernel_size=1)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
    
    def forward(self, img):
        # Preserve original input
        x = img
        # Encode
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))

        # Decode
        x = self.up(x3)
        x = torch.cat([x, x2], dim=1)
        x = self.dec2(x)
        x = self.up(x)
        x = torch.cat([x, x1], dim=1)
        x = self.dec1(x)
        
        # Produce perturbation
        delta = torch.tanh(self.out(x))
        
        # [B, 3, H, W]
        # Apply perturbation to ORIGINAL image
        return img + self.eps * delta