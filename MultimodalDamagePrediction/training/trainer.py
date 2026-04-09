import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from data.dataset import MultimodalDamageDataset
from data.augmentation import hqgan_augmentation, apply_point_cloud_augmentation
from training.losses import CompositeDamageLoss
from training.scheduler import get_scheduler
from training.monitor import TrainingMonitor
from models.image_encoder import ImageEncoder
from models.pointcloud_encoder import PointCloudEncoder
from models.metadata_encoder import MetadataEncoder
from fusion.multimodal_transformer import MultimodalTransformer
from models.decoder import CScanDecoder
import torch.nn as nn
import yaml

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

class Trainer:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = MultimodalDamageModel(self.config).to(self.device)
        self.criterion = CompositeDamageLoss().to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.config['training']['lr'], 
            weight_decay=self.config['training']['weight_decay']
        )
        self.monitor = TrainingMonitor(save_dir='checkpoints')
        
        # Mixed precision setup
        self.scaler = torch.amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu')

    def build_dataloaders(self, data_dir):
        train_dataset = MultimodalDamageDataset(data_dir, split='train', config=self.config)
        val_dataset = MultimodalDamageDataset(data_dir, split='val', config=self.config)
        
        self.train_loader = DataLoader(train_dataset, batch_size=self.config['training']['batch_size'], shuffle=True, num_workers=4)
        self.val_loader = DataLoader(val_dataset, batch_size=self.config['training']['batch_size'], shuffle=False, num_workers=4)
        
        total_steps = self.config['training']['epochs'] * len(self.train_loader)
        warmup_steps = int(total_steps * 0.1) # 10% warmup
        self.scheduler = get_scheduler(self.optimizer, warmup_steps, total_steps)

    def train_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        
        for batch in pbar:
            img = batch['image'].to(self.device)
            pc = batch['point_cloud'].to(self.device)
            meta = batch['metadata'].to(self.device)
            target = batch['cscan'].to(self.device)
            
            # Augmentation
            img = hqgan_augmentation(img)
            pc = apply_point_cloud_augmentation(pc)
            
            self.optimizer.zero_grad()
            
            # Safety check for C-Scan targets to ensure we don't train on missing data
            if target.sum() == 0 and target.shape[2] > 0:
                print("Warning: Skipped batch due to empty C-Scan target (sum=0).")
                continue
                
            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                pred = self.model(img, pc, meta)
                loss = self.criterion(pred, target)
                
            if self.monitor.check_nan(loss):
                return float('inf')
                
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            self.scheduler.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        return epoch_loss / len(self.train_loader)

    def validate(self, epoch):
        self.model.eval()
        epoch_loss = 0.0
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")
        
        with torch.no_grad():
            for batch in pbar:
                img = batch['image'].to(self.device)
                pc = batch['point_cloud'].to(self.device)
                meta = batch['metadata'].to(self.device)
                target = batch['cscan'].to(self.device)
                
                with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                    pred = self.model(img, pc, meta)
                    loss = self.criterion(pred, target)
                    
                epoch_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
                
        return epoch_loss / len(self.val_loader)

    def run(self):
        for epoch in range(1, self.config['training']['epochs'] + 1):
            train_loss = self.train_epoch(epoch)
            if math.isinf(train_loss):
                break # NaN encountered
                
            val_loss = self.validate(epoch)
            self.monitor.step(val_loss, self.model, self.optimizer, epoch)
            
            if self.monitor.early_stop:
                break
