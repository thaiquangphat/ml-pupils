import torch
import torch.nn as nn
import torch.optim as optim
from torchcrf import CRF
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
from utils.utils import get_save_name
from utils.logger import get_logger

DEFAULT_ARGS = {
    "epochs": 10,
    "batch_size": 64,
    "patience": 10,
    "split": [0.9, 0.1],
    "checkpoint_path": None,
    "lr": 0.001,
}

class CRFModel(nn.Module):
    def __init__(self, num_classes=4, feature_dim=64, input_size=(256,256)):
        super().__init__()
        self.num_classes = num_classes
        
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # (B,32,256,256)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (B,32,128,128)
            nn.Conv2d(32, feature_dim, kernel_size=3, padding=1),  # (B,64,128,128)
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (B,64,64,64)
        )
        
        Hf = input_size[0] // 4  # 256/4 = 64
        Wf = input_size[1] // 4  # 256/4 = 64
        self.num_parts = Hf * Wf  # 4096
        
        self.emission_layer = nn.Linear(feature_dim, num_classes)
        self.crf = CRF(num_tags=num_classes, batch_first=True)
        self.final_classifier = nn.Linear(num_classes, num_classes)
        
    def forward(self, x):
        B = x.size(0)
        features = self.backbone(x)  # (B,64,64,64)
        
        # reshape to (B, num_parts, feature_dim)
        features = features.permute(0, 2, 3, 1).contiguous()
        features = features.view(B, self.num_parts, -1)  # (B,4096,64)
        
        emissions = self.emission_layer(features)  # (B,4096,num_classes)
        
        # Decode with CRF (best path)
        best_paths = self.crf.decode(emissions)  # List[List[int]]
        
        # Convert to one-hot
        decoded = torch.zeros(B, self.num_parts, self.num_classes, device=x.device)
        for i, seq in enumerate(best_paths):
            for j, label in enumerate(seq):
                decoded[i, j, label] = 1.0
        
        # Aggregate (mean pooling)
        agg = decoded.mean(dim=1)  # (B,num_classes)
        
        final_logits = self.final_classifier(agg)  # (B,num_classes)
        
        return final_logits, emissions
