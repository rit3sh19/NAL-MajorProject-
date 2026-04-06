import torch
import torch.nn as nn

class MetadataEncoder(nn.Module):
    def __init__(self, input_dim=5, latent_dim=512):
        super(MetadataEncoder, self).__init__()
        
        # Input -> 5 features
        # Linear(5 -> 64)
        # ReLU
        # Linear(64 -> 256)
        # ReLU
        # Linear(256 -> 512)
        # BatchNorm + Dropout
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.Dropout(p=0.3)
        )

    def forward(self, x):
        """
        x: [B, 5]
        Returns: [B, 512]
        """
        return self.model(x)
