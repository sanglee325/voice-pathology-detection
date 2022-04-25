import torch
import torch.nn as nn
import torch.optim as optim

class CNNNetwork(nn.Module):
    
    def __init__(self, num_classes=1):
                  
        super().__init__()
        
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding = (3, 3)),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Dropout(p = 0.2),
            
            #layernorm & GeLu
            nn.Conv2d(16, 64, kernel_size=3, stride=2, padding = 1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Dropout(p = 0.2),
            nn.MaxPool2d(kernel_size=3, stride=2), #maxpool location switching
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding = 1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Dropout(p = 0.2),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding = 1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten()
            )
        
        self.cls_layer = nn.Sequential(
            nn.Linear(256, num_classes),
            nn.Sigmoid()
            )
    
    def forward(self, x, return_feats=False):

        feats = self.backbone(x)
        out = self.cls_layer(feats)

        if return_feats:
            return feats
        else:
            return out
