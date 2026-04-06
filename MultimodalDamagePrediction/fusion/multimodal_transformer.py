import torch
import torch.nn as nn

class MultimodalTransformer(nn.Module):
    def __init__(self, embed_dim=512, num_layers=4, num_heads=8):
        super(MultimodalTransformer, self).__init__()
        
        # We define a class token specifically if we want, but since we flatten 
        # all 3 modalities, we can just use the transformer to mix them and flatten.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Positional encodings for the 3 modalities (Image, PC, Meta)
        self.pos_embedding = nn.Parameter(torch.randn(1, 3, embed_dim))

    def forward(self, img_feat, pc_feat, meta_feat):
        """
        img_feat: [B, 512]
        pc_feat: [B, 512]
        meta_feat: [B, 512]
        """
        # Stack -> [B, 3, 512]
        x = torch.stack([img_feat, pc_feat, meta_feat], dim=1)
        
        # Add position embeddings
        x = x + self.pos_embedding
        
        # Transformer mixing
        x = self.transformer(x)
        
        # Output [B, 1536]
        x = x.flatten(start_dim=1)
        return x
