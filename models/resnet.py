import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as ttf

class BinaryResNet18(nn.Module):
    
    def __init__(self, num_classes=1, pretrained=False):
                  
        super().__init__()
        
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            torchvision.models.resnet18(pretrained=pretrained)
            #torchvision.models.resnet18()
            )
        
        self.cls_layer = nn.Sequential(
            nn.Identity(),
            nn.Linear(1000, num_classes),
            nn.Sigmoid()
            )
    
    def forward(self, x, return_feats=False):

        feats = self.backbone(x)
        out = self.cls_layer(feats)

        if return_feats:
            return feats
        else:
            return out