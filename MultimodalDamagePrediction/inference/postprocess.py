import cv2
import numpy as np
import torch

def generate_heatmap(cscan_tensor):
    """
    Given a [1, H, W] tensor in range [0, 1], generates an RGB heatmap of damage severity.
    """
    cscan_np = cscan_tensor.squeeze(0).cpu().numpy()
    cscan_norm = (cscan_np * 255).astype(np.uint8)
    
    # Apply JET colormap
    heatmap = cv2.applyColorMap(cscan_norm, cv2.COLORMAP_JET)
    
    # OpenCV uses BGR, convert to RGB
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    return heatmap_rgb

def calculate_severity_score(cscan_tensor):
    """
    Calculates a severity score [0-100] based on the mean pixel intensity 
    of the top 5% most intense pixels in the predicted c-scan.
    """
    cscan_flat = cscan_tensor.view(-1)
    
    k = max(1, int(0.05 * cscan_flat.size(0)))
    topk_vals = torch.topk(cscan_flat, k)[0]
    score = topk_vals.mean().item() * 100
    
    return min(max(int(score), 0), 100)
