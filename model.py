import torch
import torch.nn as nn
from torchvision.models import resnet18

class ResNet18(nn.Module):
    def __init__(self, n_base_classes=64):
        super(ResNet18, self).__init__()
        
        base_model = resnet18(weights='IMAGENET1K_V1')
        
        self.encoder = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
            base_model.layer1,
            base_model.layer2,
            base_model.layer3,
            base_model.layer4,
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.feature_dim = 512
        
        self.classifier = nn.Linear(self.feature_dim, n_base_classes)

    def forward(self, x, return_features=False):
        features = self.encoder(x)
        features = features.view(features.size(0), -1)
        
        if return_features:
            return features
            
        logits = self.classifier(features)
        return logits

if __name__ == '__main__':
    model = ResNet18(n_base_classes=64)
    test_image = torch.randn(4, 3, 84, 84)
    
    logits = model(test_image)
    print(f"Output logits shape: {logits.shape}")
    
    features = model(test_image, return_features=True)
    print(f"Output features shape: {features.shape}")