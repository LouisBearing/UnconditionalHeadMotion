import torch
import torch.nn as nn
import numpy as np
from .networks import *

# Source: https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/vggm.py

class SpatialCrossMapLRN(nn.Module):
    def __init__(self, local_size=1, alpha=1.0, beta=0.75, k=1, ACROSS_CHANNELS=True):
        super(SpatialCrossMapLRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if ACROSS_CHANNELS:
            self.average=nn.AvgPool3d(kernel_size=(local_size, 1, 1),
                    stride=1,
                    padding=(int((local_size-1.0)/2), 0, 0))
        else:
            self.average=nn.AvgPool2d(kernel_size=local_size,
                    stride=1,
                    padding=int((local_size-1.0)/2))
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(self.k).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(self.k).pow(self.beta)
        x = x.div(div)
        return x

class VGGM(nn.Module):

    def __init__(self, num_classes=1000, self_attention=False):
        super(VGGM, self).__init__()
        self.num_classes = num_classes
        features = [
            nn.Conv2d(1,96,(7, 5),(2, 1), (0, 2)),
            nn.ReLU(),
            SpatialCrossMapLRN(5, 0.0005, 0.75, 2),
            nn.MaxPool2d((3, 3),(2, 1),(0, 0),ceil_mode=True),
            nn.Conv2d(96,256,(5, 5),(2, 2),(1, 1)),
            nn.ReLU(),
            SpatialCrossMapLRN(5, 0.0005, 0.75, 2),
            nn.MaxPool2d((3, 3),(2, 2),(0, 0),ceil_mode=True),
            nn.Conv2d(256,512,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            # nn.MaxPool2d((3, 3),(2, 2),(0, 0),ceil_mode=True)
        ]
        if self_attention:
            features.insert(2, MultiHeadAttention(96, 64, 1))
        self.features = nn.Sequential(*features)
        self.classif = nn.Sequential(
            nn.Linear(3584,4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096,num_classes)
        )

    def forward(self, x):
        # Expected input shape: bs, 128, 20

        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classif(x)
        return x