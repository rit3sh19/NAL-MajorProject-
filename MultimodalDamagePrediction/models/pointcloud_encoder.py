import torch
import torch.nn as nn
import torch.nn.functional as F

class PointCloudEncoder(nn.Module):
    def __init__(self, latent_dim=512):
        super(PointCloudEncoder, self).__init__()
        # PyTorch native proxy for PointNet/Point Transformer
        # This operates on [B, 3, N]
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 1024, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, latent_dim)
        
        self.bn5 = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        """
        x: [B, N, 3] usually, but Conv1d needs [B, C, N]
        Returns: [B, 512]
        """
        x = x.transpose(1, 2)  # [B, 3, N]
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        # Max pooling over points
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        
        x = F.relu(self.bn5(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
