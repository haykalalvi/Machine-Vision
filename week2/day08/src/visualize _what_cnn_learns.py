import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# =============================================
# LOAD A PRETRAINED MODEL AND INSPECT ITS FILTERS
# We use ResNet18 trained on ImageNet (1.2M images)
# These filters are what "good" CNN features look like
# =============================================

model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
model.eval()

# Get the first convolutional layer's weights
# Shape: [out_channels, in_channels, kernel_h, kernel_w]
# = [64, 3, 7, 7] for ResNet18's first layer
first_conv_weights = model.conv1.weight.data.numpy()

print(f"First conv layer: {first_conv_weights.shape}")
print(f"= {first_conv_weights.shape[0]} filters, each is {first_conv_weights.shape[2]}x{first_conv_weights.shape[3]} with {first_conv_weights.shape[1]} color channels")

# Visualize the 64 learned filters
fig, axes = plt.subplots(8, 8, figsize=(14, 14))
fig.suptitle('ResNet18 First Layer: 64 Learned Filters\n'
             'Each filter is a 7×7 pattern detector — the network learned these from 1.2M images',
             fontsize=12, fontweight='bold')

for idx, ax in enumerate(axes.flat):
    if idx < 64:
        # Each filter has 3 channels (RGB) — normalize to display
        f = first_conv_weights[idx]
        f = (f - f.min()) / (f.max() - f.min())
        f = np.transpose(f, (1, 2, 0))  # [C,H,W] → [H,W,C]
        ax.imshow(f)
        ax.axis('off')
        ax.set_title(f'F{idx}', fontsize=7)

plt.tight_layout()
plt.savefig('output/03a_resnet_learned_filters.jpg', dpi=150)
plt.show()

print("\nWhat do you notice about these filters?")
print("→ Many look like edge detectors (horizontal, vertical, diagonal)")
print("→ Some look like color blob detectors")
print("→ This matches what neuroscience knows about the human visual cortex V1")
print("→ These same patterns emerge regardless of dataset — they are universal")

# =============================================
# TRAIN A TINY CNN ON MNIST AND WATCH WHAT IT LEARNS
# Compare random init vs after training
# =============================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_data = torchvision.datasets.MNIST('./data', train=True,  download=True, transform=transform)
test_data  = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=256, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_data,  batch_size=256)

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.relu  = nn.ReLU()
        self.fc    = nn.Linear(16 * 7 * 7, 10)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_mnist = SmallCNN().to(device)

def get_first_layer_filters(model):
    """Extract and normalize first conv layer filters for visualization"""
    w = model.conv1.weight.data.cpu().numpy()  # [8, 1, 5, 5]
    return w

def visualize_filters(filters_before, filters_after, save_path):
    fig, axes = plt.subplots(2, 8, figsize=(18, 5))
    fig.suptitle('CNN First Layer Filters: Random Init vs After Training\n'
                 'Watch the filters become meaningful feature detectors',
                 fontsize=12, fontweight='bold')
    
    for i in range(8):
        # Before training
        f_before = filters_before[i, 0]
        f_before = (f_before - f_before.min()) / (f_before.max() - f_before.min() + 1e-8)
        axes[0][i].imshow(f_before, cmap='RdBu_r')
        axes[0][i].set_title(f'Filter {i+1}', fontsize=9)
        axes[0][i].axis('off')
        if i == 0:
            axes[0][i].set_ylabel('Before\ntraining', fontsize=10, fontweight='bold')
        
        # After training
        f_after = filters_after[i, 0]
        f_after = (f_after - f_after.min()) / (f_after.max() - f_after.min() + 1e-8)
        axes[1][i].imshow(f_after, cmap='RdBu_r')
        axes[1][i].axis('off')
        if i == 0:
            axes[1][i].set_ylabel('After\ntraining', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()

# Save filters before training
filters_before = get_first_layer_filters(model_mnist)

# Train for 3 epochs
optimizer = torch.optim.Adam(model_mnist.parameters(), lr=1e-3) #the adjusters to lower the loss function
criterion = nn.CrossEntropyLoss() # loss functions (the grader)

print("\nTraining CNN on MNIST for 3 epochs...")
train_losses, test_accs = [], []

for epoch in range(3):
    model_mnist.train()
    running_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model_mnist(imgs), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    # Evaluate
    model_mnist.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model_mnist(imgs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    
    acc = correct / total * 100
    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)
    test_accs.append(acc)
    print(f"  Epoch {epoch+1}/3 — Loss: {avg_loss:.4f} | Test Acc: {acc:.2f}%")

# Save filters after training and compare
filters_after = get_first_layer_filters(model_mnist)
visualize_filters(filters_before, filters_after, 'output/03b_filters_before_after.jpg')

print("\n→ Notice: before training, filters look like random noise")
print("→ After training, filters develop structure — edge detectors, blob detectors")
print("→ This is exactly what happens in YOUR defect detection model")