import torch
import torch.nn as nn
from torchvision.models import resnet18

class BaseResNet18(nn.Module):
    def __init__(self, num_classes=64):
        super(BaseResNet18, self).__init__()
        self.backbone = resnet18(pretrained=False)
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity() # Remove initial maxpool
        self.feature_dim = 512
        self.backbone.fc = nn.Identity()
        self.fc = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        features = x.view(x.size(0), -1)
        out = self.fc(features)
        return out, features