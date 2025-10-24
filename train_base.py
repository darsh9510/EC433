import torch
import torch.optim as optim
import torch.utils.data as data
from model import ResNet18
from utils import LabelSmoothingLoss

N_BASE_CLASSES = 64
N_EPOCHS = 90
BATCH_SIZE = 128
LEARNING_RATE = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_stub_dataloader(n_classes, batch_size):
    print("Using STUB dataloader for base training...")
    n_samples = batch_size * 1000
    images = torch.randn(n_samples, 3, 84, 84)
    labels = torch.randint(0, n_classes, (n_samples,))
    dataset = data.TensorDataset(images, labels)
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

def main():
    print(f"Starting base model training on {DEVICE}...")
    
    model = ResNet18(n_base_classes=N_BASE_CLASSES).to(DEVICE)
    model.train()
    
    criterion = LabelSmoothingLoss(n_classes=N_BASE_CLASSES, smoothing=0.1).to(DEVICE)
    
    optimizer = optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )
    
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[45, 66], gamma=0.1
    )
    
    train_loader = get_stub_dataloader(N_BASE_CLASSES, BATCH_SIZE)
    
    for epoch in range(N_EPOCHS):
        total_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            logits = model(images)
            
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if i > 10:
                break
                
        scheduler.step()
        avg_loss = total_loss / (i + 1)
        print(f"Epoch {epoch+1}/{N_EPOCHS} | Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.5f}")

    torch.save(model.state_dict(), 'base_model.pth')
    print("Base model training complete. Saved to 'base_model.pth'")

if __name__ == '__main__':
    main()