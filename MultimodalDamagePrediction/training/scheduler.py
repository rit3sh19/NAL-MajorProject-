import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

def get_scheduler(optimizer, warmup_steps, total_steps):
    """
    Returns Cosine Annealing with Warmup.
    """
    # Linear warmup
    warmup_lrs = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    
    # Cosine Annealing
    cosine_lrs = CosineAnnealingLR(optimizer, T_max=(total_steps - warmup_steps))
    
    # Sequence of two
    scheduler = SequentialLR(optimizer, schedulers=[warmup_lrs, cosine_lrs], milestones=[warmup_steps])
    
    return scheduler
