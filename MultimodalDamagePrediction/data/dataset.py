import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import torchvision.transforms as T
import os
import json

class MultimodalDamageDataset(Dataset):
    """
    Multimodal Dataset for Aircraft Composite Internal Damage Predictor.
    Loads Image, Point Cloud, Metadata, and corresponding C-Scan targets.
    """
    def __init__(self, data_dir, split='train', config=None):
        self.data_dir = data_dir
        self.split = split
        self.config = config
        self.samples = self._load_samples()
        
        self.img_transform = T.Compose([
            T.Resize((self.config['dataset']['resize'][0], self.config['dataset']['resize'][1])),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _load_samples(self):
        """
        Parses a metadata/manifest file to gather samples.
        Assuming structured dataset directory or manifest file.
        For production, read from data_dir/split_manifest.json
        """
        manifest_path = os.path.join(self.data_dir, f"{self.split}_manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path, 'r') as f:
                return json.load(f)
        return []

    def __len__(self):
        return len(self.samples)

    def _process_point_cloud(self, pc_path):
        """
        Loads point cloud file (PCD/PLY).
        Requires Voxel grid downsampling, MLS smoothing, Normal + curvature estimation (In external preprocessing).
        Here we load the processed N x 3 data.
        """
        # Missing point cloud -> zero tensor
        if pc_path is None or not os.path.exists(pc_path):
            return torch.zeros((self.config['dataset']['point_cloud_points'], 3))
        
        try:
            # We assume point clouds have been preprocessed and saved as .npy [N, 3]
            points = np.load(pc_path)
            # Sample exactly fixed number of points
            num_points = self.config['dataset']['point_cloud_points']
            if points.shape[0] >= num_points:
                indices = np.random.choice(points.shape[0], num_points, replace=False)
            else:
                indices = np.random.choice(points.shape[0], num_points, replace=True)
            points = points[indices, :]
            return torch.tensor(points, dtype=torch.float32)
        except Exception:
            # Missing/Corrupted point cloud handling -> zero tensor
            return torch.zeros((self.config['dataset']['point_cloud_points'], 3))
        
    def _process_metadata(self, meta_dict):
        """
        Process user metadata inputs
        Features: Dent depth, Damage area, Thickness, Layup sequence, Material type
        Missing metadata -> mean imputation
        """
        # Defaults acts as mean imputation/baselines for mock demo. 
        # In a real system, you'd calculate column means from the training set.
        features = [
            float(meta_dict.get('dent_depth', 0.0)),
            float(meta_dict.get('damage_area', 0.0)),
            float(meta_dict.get('thickness', 0.0)),
            float(meta_dict.get('layup_sequence_encoded', 0.0)), # Should be label encoded
            float(meta_dict.get('material_type_encoded', 0.0)) # CFRP=0, GFRP=1, Hybrid=2
        ]
        return torch.tensor(features, dtype=torch.float32)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        
        # 1. Image [3, 224, 224]
        img_path = sample_info.get('image_path')
        if img_path and os.path.exists(img_path):
            img = Image.open(img_path).convert('RGB')
            img_tensor = self.img_transform(img)
        else:
            # Missing image -> zero tensor
            img_tensor = torch.zeros((3, *self.config['dataset']['resize']))
            
        # 2. Point Cloud [N, 3]
        pc_tensor = self._process_point_cloud(sample_info.get('pc_path'))
        
        # 3. Metadata [5]
        meta_tensor = self._process_metadata(sample_info.get('metadata', {}))
        
        # 4. Target C-Scan [1, 256, 256]
        cscan_path = sample_info.get('cscan_path')
        cscan_size = self.config['dataset']['cscan_size']
        if self.split in ['train', 'val'] and cscan_path and os.path.exists(cscan_path):
            cscan_img = Image.open(cscan_path).convert('L')
            cscan_img = cscan_img.resize(tuple(cscan_size))
            cscan_tensor = T.ToTensor()(cscan_img)
        else:
            cscan_tensor = torch.zeros((1, *cscan_size))

        return {
            "image": img_tensor,
            "point_cloud": pc_tensor,
            "metadata": meta_tensor,
            "cscan": cscan_tensor,
            "sample_id": sample_info.get('id', f'sample_{idx}')
        }
