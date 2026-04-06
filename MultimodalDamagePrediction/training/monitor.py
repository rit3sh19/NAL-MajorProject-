import torch
import os
import math

class TrainingMonitor:
    """
    Handles stability requirements:
    - Early stopping
    - Checkpoint saving
    - NaN detection
    """
    def __init__(self, patience=10, save_dir='checkpoints'):
        self.patience = patience
        self.save_dir = save_dir
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False
        
        os.makedirs(save_dir, exist_ok=True)

    def check_nan(self, loss):
        if math.isnan(loss.item()):
            print("NaN detected in loss! Stopping training.")
            return True
        return False

    def step(self, val_loss, model, optimizer, epoch):
        is_best = val_loss < self.best_loss
        if is_best:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss
            }, 'best_model.pth')
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"Early stopping triggered at epoch {epoch}")
                
        # Save latest
        self.save_checkpoint({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss
        }, 'latest_model.pth')

    def save_checkpoint(self, state, filename):
        path = os.path.join(self.save_dir, filename)
        torch.save(state, path)
