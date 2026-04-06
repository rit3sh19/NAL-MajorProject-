import torch
import torch.nn as nn
import torch.nn.functional as F

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        # Using Upsample + Conv instead of ConvTranspose to reduce checkerboard artifacts
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x

class CScanDecoder(nn.Module):
    def __init__(self, input_dim=1536):
        super(CScanDecoder, self).__init__()
        
        # 1. Project latent code to 256 * 4 * 4
        self.proj = nn.Linear(input_dim, 256 * 4 * 4)
        
        # 4x4 -> 8x8
        self.up1 = UpBlock(256, 128)
        # 8x8 -> 16x16
        self.up2 = UpBlock(128, 64)
        # 16x16 -> 32x32
        self.up3 = UpBlock(64, 32)
        # 32x32 -> 64x64
        self.up4 = UpBlock(32, 16)
        # 64x64 -> 128x128
        self.up5 = UpBlock(16, 8)
        # 128x128 -> 256x256
        self.up6 = UpBlock(8, 8)
        
        # Output to 1 channel (grayscale C-Scan)
        self.final_conv = nn.Conv2d(8, 1, kernel_size=3, padding=1)

    def forward(self, x):
        """
        x: [B, 1536]
        """
        x = self.proj(x)
        x = x.view(-1, 256, 4, 4)
        
        x = self.up1(x)  # [B, 128, 8, 8]
        x = self.up2(x)  # [B, 64, 16, 16]
        x = self.up3(x)  # [B, 32, 32, 32]
        x = self.up4(x)  # [B, 16, 64, 64]
        x = self.up5(x)  # [B, 8, 128, 128]
        x = self.up6(x)  # [B, 8, 256, 256]
        
        x = self.final_conv(x) # [B, 1, 256, 256]
        # Depending on targets, Sigmoid or raw logits
        # For MSE+SSIM+Dice, target is usually 0-1 range.
        return torch.sigmoid(x)
