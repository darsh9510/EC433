import torch
import torch.optim as optim
from model import BaseResNet18
from utils import get_data_loader, LabelSmoothingLoss
import os

DATA_ROOT = "./data/mini_imagenet"
EPOCHS = 90
LR = 0.1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_base():
    print("Starting Base Training...")
    loader = get_data_loader(DATA_ROOT, mode='train')
    model = BaseResNet18(num_classes=64).to(DEVICE)
    
    criterion = LabelSmoothingLoss(smoothing=0.1)
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[45, 66], gamma=0.1)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs, _ = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        scheduler.step()
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), "base_model.pth")
    print("Base model saved.")

if __name__ == "__main__":
    train_base()