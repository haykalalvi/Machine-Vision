import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_auc_score, recall_score
import wandb

# =============================================
# CONFIG — change these to experiment
# =============================================
CONFIG = {
    'epochs':       10,
    'batch_size':   64,
    'lr':           1e-3,
    'weight_decay': 1e-2,
    'num_classes':  10,
    'device':       'cuda' if torch.cuda.is_available() else 'cpu',
}

print(f"Training on: {CONFIG['device']}")

# =============================================
# INITIALIZE WANDB — experiment tracking
# Every run is saved and comparable
# =============================================
wandb.init(
    project="day09-training-discipline",
    config=CONFIG,
    name=f"cnn-cifar10-adamw-lr{CONFIG['lr']}"
)

# =============================================
# DATA — CIFAR-10 with augmentation
# =============================================
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

train_data = torchvision.datasets.CIFAR10('./data', train=True,  download=True, transform=train_transform)
test_data  = torchvision.datasets.CIFAR10('./data', train=False, download=True, transform=test_transform)

train_loader = DataLoader(train_data, batch_size=CONFIG['batch_size'], shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_data,  batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)

print(f"Training samples: {len(train_data)}")
print(f"Test samples:     {len(test_data)}")

# =============================================
# MODEL — same TinyCNN from Day 8, now with more filters
# =============================================
class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.1),

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.2),

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.3),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

model     = CNN(CONFIG['num_classes']).to(CONFIG['device'])
optimizer = torch.optim.AdamW(model.parameters(),
                               lr=CONFIG['lr'],
                               weight_decay=CONFIG['weight_decay'])
criterion = nn.CrossEntropyLoss()

# Learning rate scheduler — reduces LR when plateau is detected
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=CONFIG['lr'],
    steps_per_epoch=len(train_loader),
    epochs=CONFIG['epochs']
)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")
wandb.watch(model, log_freq=100)

# =============================================
# TRAINING LOOP — production style
# =============================================
best_acc  = 0
train_losses, test_accs = [], []

for epoch in range(CONFIG['epochs']):
    # --- TRAIN ---
    model.train()
    running_loss, n_correct, n_total = 0, 0, 0

    for batch_idx, (imgs, labels) in enumerate(train_loader):
        imgs, labels = imgs.to(CONFIG['device']), labels.to(CONFIG['device'])

        optimizer.zero_grad()
        outputs = model(imgs)
        loss    = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping — prevents exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        n_correct    += (outputs.argmax(1) == labels).sum().item()
        n_total      += labels.size(0)

    train_loss = running_loss / len(train_loader)
    train_acc  = n_correct / n_total

    # --- EVALUATE ---
    model.eval()
    all_preds, all_labels_eval = [], []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(CONFIG['device']), labels.to(CONFIG['device'])
            outputs = model(imgs)
            preds   = outputs.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels_eval.extend(labels.cpu().numpy())

    all_preds       = np.array(all_preds)
    all_labels_eval = np.array(all_labels_eval)

    test_acc = (all_preds == all_labels_eval).mean()
    f1       = f1_score(all_labels_eval, all_preds, average='macro')
    current_lr = scheduler.get_last_lr()[0]

    # Save best model
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), 'output/best_model.pt')

    # Log to W&B
    wandb.log({
        'epoch':      epoch + 1,
        'train/loss': train_loss,
        'train/acc':  train_acc,
        'test/acc':   test_acc,
        'test/f1':    f1,
        'lr':         current_lr,
    })

    train_losses.append(train_loss)
    test_accs.append(test_acc)

    print(f"Epoch {epoch+1:2d}/{CONFIG['epochs']} | "
          f"Loss: {train_loss:.4f} | "
          f"Train: {train_acc:.3f} | "
          f"Test: {test_acc:.3f} | "
          f"F1: {f1:.3f} | "
          f"LR: {current_lr:.6f} | "
          f"{'✓ Best' if test_acc == best_acc else ''}")

print(f"\nBest test accuracy: {best_acc:.4f}")
wandb.finish()

# Plot training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
ax1.plot(train_losses, 'b-o', linewidth=2, markersize=5, label='Train loss')
ax1.set_title('Training Loss')
ax1.set_xlabel('Epoch')
ax1.grid(True, alpha=0.3)
ax1.legend()

ax2.plot([a*100 for a in test_accs], 'g-o', linewidth=2, markersize=5, label='Test accuracy')
ax2.set_title(f'Test Accuracy (Best: {best_acc*100:.2f}%)')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig('output/04_training_curves.jpg', dpi=150)
plt.show()