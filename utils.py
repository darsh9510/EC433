import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np

ALPHA = 0.1
LAMBDA = 0.1
TEMP = 15.0

class MiniImageNet(Dataset):
    def __init__(self, root, mode='train', transform=None):
        self.transform = transform
        self.mode = mode
        self.root = root
        
        all_classes = sorted(os.listdir(root))
        if mode == 'train':
            self.classes = all_classes[:64]
        elif mode == 'test':
            self.classes = all_classes[80:]
        
        self.data = []
        self.targets = []
        
        for i, cls in enumerate(self.classes):
            cls_path = os.path.join(root, cls)
            imgs = [os.path.join(cls_path, x) for x in os.listdir(cls_path)]
            self.data.extend(imgs)
            self.targets.extend([i] * len(imgs))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = self.data[idx]
        label = self.targets[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

def get_data_loader(root, mode='train', batch_size=128):
    if mode == 'train':
        transform = transforms.Compose([
            transforms.Resize(92),
            transforms.RandomResizedCrop(84),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(92),
            transforms.CenterCrop(84),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    dataset = MiniImageNet(root, mode=mode, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=(mode=='train'), num_workers=4)
    return loader

class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        confidence = 1.0 - self.smoothing
        log_probs = F.log_softmax(pred, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

def get_metric_similarity(features, weights):
    features = F.normalize(features, p=2, dim=1)
    weights = F.normalize(weights, p=2, dim=1)
    logits = TEMP * torch.mm(features, weights.t())
    return logits