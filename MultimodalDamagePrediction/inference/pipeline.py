import torch
from torchvision import transforms as T
from PIL import Image
import yaml
import os
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.pointcloud_io import load_point_cloud

from models.image_encoder import ImageEncoder
from models.pointcloud_encoder import PointCloudEncoder
from models.metadata_encoder import MetadataEncoder
from fusion.multimodal_transformer import MultimodalTransformer
from models.decoder import CScanDecoder
import torch.nn as nn

class MultimodalDamageModel(nn.Module):
    def __init__(self, config):
        super(MultimodalDamageModel, self).__init__()
        self.img_enc = ImageEncoder(latent_dim=config['model']['latent_dim'])
        self.pc_enc = PointCloudEncoder(latent_dim=config['model']['latent_dim'])
        self.meta_enc = MetadataEncoder(input_dim=config['model']['metadata_dim'], latent_dim=config['model']['latent_dim'])
        self.fusion = MultimodalTransformer(
            embed_dim=config['model']['latent_dim'], 
            num_layers=config['model']['transformer_layers'], 
            num_heads=config['model']['transformer_heads']
        )
        self.decoder = CScanDecoder(input_dim=config['model']['merged_dim'])

    def forward(self, img, pc, meta):
        img_feat = self.img_enc(img)
        pc_feat = self.pc_enc(pc)
        meta_feat = self.meta_enc(meta)
        
        fused = self.fusion(img_feat, pc_feat, meta_feat)
        out = self.decoder(fused)
        return out


class InferencePipeline:
    def __init__(self, config_path, checkpoint_path=None):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = MultimodalDamageModel(self.config).to(self.device)
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print("Loaded checkpoint successfully.")
        else:
            print("Warning: No checkpoint loaded. Output will be random.")
            
        self.model.eval()

        self.img_transform = T.Compose([
            T.Resize((self.config['dataset']['resize'][0], self.config['dataset']['resize'][1])),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def preprocess_image(self, image_input):
        if isinstance(image_input, str): # path
            img = Image.open(image_input).convert('RGB')
        else:
            img = image_input.convert('RGB')
        return self.img_transform(img).unsqueeze(0).to(self.device)

    def preprocess_pointcloud(self, pc_path):
        num_points = self.config['dataset']['point_cloud_points']
        if pc_path is None or not os.path.exists(pc_path):
            return torch.zeros((1, num_points, 3)).to(self.device)
        
        try:
            # Supports .ply and .npy via shared loader
            points = load_point_cloud(pc_path)  # [N, 3] float32
            if points.shape[0] >= num_points:
                indices = np.random.choice(points.shape[0], num_points, replace=False)
            else:
                indices = np.random.choice(points.shape[0], num_points, replace=True)
            points = points[indices, :]
            return torch.tensor(points, dtype=torch.float32).unsqueeze(0).to(self.device)
        except Exception as e:
            print(f"Warning: Could not load point cloud '{pc_path}': {e}")
            return torch.zeros((1, num_points, 3)).to(self.device)

    def preprocess_metadata(self, meta_dict):
        features = [
            float(meta_dict.get('dent_depth', 0.0)),
            float(meta_dict.get('damage_area', 0.0)),
            float(meta_dict.get('thickness', 0.0)),
            float(meta_dict.get('layup_sequence_encoded', 0.0)), 
            float(meta_dict.get('material_type_encoded', 0.0)) 
        ]
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(self, image_input, pc_path, metadata):
        img_tensor = self.preprocess_image(image_input)
        pc_tensor = self.preprocess_pointcloud(pc_path)
        meta_tensor = self.preprocess_metadata(metadata)
        
        output = self.model(img_tensor, pc_tensor, meta_tensor)
        return output[0] # Return the [1, 256, 256] tensor instead of batch
