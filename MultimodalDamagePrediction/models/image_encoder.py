import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

class ImageEncoder(nn.Module):
    def __init__(self, latent_dim=512):
        super(ImageEncoder, self).__init__()
        # Use Vision Transformer (ViT-Base)
        self.vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        
        # Original ViT outputs 1000 classes because of ImageNet.
        # We replace the final head with our projection to [B, 512]
        in_features = self.vit.heads.head.in_features # 768 for vit_b_16
        
        self.vit.heads = nn.Sequential(
            nn.Linear(in_features, latent_dim)
        )
        
    def forward(self, x):
        """
        x: [B, 3, 224, 224]
        Returns: [B, 512]
        """
        return self.vit(x)
