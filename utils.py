import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothingLoss(nn.Module):
    def __init__(self, n_classes, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.n_classes = n_classes

    def forward(self, logits, targets):
        with torch.no_grad():
            smooth_labels = torch.full_like(logits, self.smoothing / (self.n_classes - 1))
            smooth_labels.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        
        log_probs = F.log_softmax(logits, dim=1)
        loss = (-smooth_labels * log_probs).sum(dim=1).mean()
        return loss

def calculate_prototypes(features, labels, k_way):
    prototypes = torch.zeros(k_way, features.shape[1], device=features.device)
    for i in range(k_way):
        class_features = features[labels == i]
        prototypes[i] = class_features.mean(dim=0)
    return prototypes

def conditional_entropy(logits):
    probs = F.softmax(logits, dim=1)
    log_probs = F.log_softmax(logits, dim=1)
    loss = -(probs * log_probs).sum(dim=1).mean()
    return loss

def marginal_entropy(logits):
    mean_probs = F.softmax(logits, dim=1).mean(dim=0)
    loss = -(mean_probs * torch.log(mean_probs + 1e-8)).sum()
    return loss