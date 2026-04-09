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

    def forward_with_features(self, x):
        """
        Explicitly extract feature maps from ViT before final projection.
        x: [B, 3, 224, 224]
        Returns: 
           latent: [B, 512]
           spatial_features: [B, 768, 14, 14] 
                 (The 196 patch tokens reshaped back into a 14x14 grid)
        """
        # Patch embedding and pos embedding
        x = self.vit._process_input(x)
        
        # Expand class token
        n = x.shape[0]
        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        
        # Pass through transformer encoder layers
        x = self.vit.encoder(x)
        
        # Class token is at index 0, patches are at index 1 to 196
        cls_token = x[:, 0]
        patch_tokens = x[:, 1:]
        
        # Reshape patch tokens to spatial feature map (14x14)
        B, N, C = patch_tokens.shape # N=196, C=768
        spatial_features = patch_tokens.transpose(1, 2).contiguous().view(B, C, 14, 14)
        
        # Final projection on cls token
        latent = self.vit.heads(cls_token)
        return latent, spatial_features
