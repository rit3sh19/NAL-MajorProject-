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
        
        x = x.flatten(start_dim=1)
        return x

    def forward_with_attention(self, img_feat, pc_feat, meta_feat):
        """
        Returns the output along with intermediate layer outputs and attention matrices for visualization.
        """
        x = torch.stack([img_feat, pc_feat, meta_feat], dim=1)
        x = x + self.pos_embedding
        
        layer_outputs = []
        attn_maps = []
        
        x_iter = x
        for layer in self.transformer.layers:
            # We must replicate the forward pass of nn.TransformerEncoderLayer 
            # to capture the attention weights (need_weights=True)
            attn_input = layer.norm1(x_iter) if layer.norm_first else x_iter
            attn_output, attn_weights = layer.self_attn(
                attn_input, attn_input, attn_input, need_weights=True
            )
            
            # The rest of the TransformerEncoderLayer (assuming not norm_first for standard PyTorch)
            if not layer.norm_first:
                x_iter = layer.norm1(x_iter + layer.dropout1(attn_output))
                x_iter = layer.norm2(x_iter + layer.dropout2(layer.linear2(layer.dropout(layer.activation(layer.linear1(x_iter))))))
            else:
                x_iter = x_iter + layer.dropout1(attn_output)
                x_iter = x_iter + layer.dropout2(layer.linear2(layer.dropout(layer.activation(layer.linear1(layer.norm2(x_iter))))))
                
            layer_outputs.append(x_iter.clone())
            attn_maps.append(attn_weights.clone())
            
        fused = x_iter.flatten(start_dim=1)
        return fused, layer_outputs, attn_maps
