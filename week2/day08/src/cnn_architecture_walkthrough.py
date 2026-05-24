import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# =============================================
# BUILD A CNN AND WATCH WHAT HAPPENS TO THE
# TENSOR SHAPE AT EVERY SINGLE LAYER
# =============================================

class TinyCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        
        # Block 1: detect simple features (edges, textures)
        self.block1 = nn.Sequential(
            # Input:  [batch, 1, 64, 64]   (grayscale 64x64 image)
            # Output: [batch, 16, 64, 64]  (16 feature maps, same spatial size)
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # Output: [batch, 16, 32, 32]  (halved by pooling)
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 2: detect combinations of simple features
        self.block2 = nn.Sequential(
            # Input:  [batch, 16, 32, 32]
            # Output: [batch, 32, 32, 32]
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # Output: [batch, 32, 16, 16]
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 3: detect higher-level patterns
        self.block3 = nn.Sequential(
            # Input:  [batch, 32, 16, 16]
            # Output: [batch, 64, 16, 16]
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # Output: [batch, 64, 8, 8]
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Classifier: flatten and decide the class
        self.classifier = nn.Sequential(
            # Flatten: [batch, 64, 8, 8] → [batch, 64*8*8] = [batch, 4096]
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.5),             # randomly zero 50% of neurons during training
            nn.Linear(128, num_classes)  # final output: [batch, num_classes]
        )
    
    def forward(self, x):
        print(f"  Input:   {tuple(x.shape)}")
        x = self.block1(x)
        print(f"  Block 1: {tuple(x.shape)}  ← 16 feature maps, spatial halved")
        x = self.block2(x)
        print(f"  Block 2: {tuple(x.shape)}  ← 32 feature maps, spatial halved again")
        x = self.block3(x)
        print(f"  Block 3: {tuple(x.shape)}  ← 64 feature maps, spatial halved again")
        x = self.classifier(x)
        print(f"  Output:  {tuple(x.shape)}  ← raw scores (logits) per class")
        return x

# Create model and run a dummy image through it
model = TinyCNN(num_classes=2)
dummy_image = torch.randn(1, 1, 64, 64)  # batch=1, channels=1, height=64, width=64

print("=" * 55)
print("FORWARD PASS — watching tensor shape at every layer")
print("=" * 55)
with torch.no_grad():
    output = model(dummy_image)

print(f"\nFinal output (raw logits): {output}")
print(f"After softmax (probabilities): {torch.softmax(output, dim=1)}")

# =============================================
# COUNT PARAMETERS — understand model size
# =============================================
total_params = sum(p.numel() for p in model.parameters())
trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\n=== Model Size ===")
print(f"Total parameters:     {total_params:,}")
print(f"Trainable parameters: {trainable:,}")
print(f"Memory (float32):     ~{total_params * 4 / 1024:.1f} KB")

# =============================================
# VISUALIZE: what does BatchNorm actually do?
# =============================================
print("\n=== Understanding BatchNorm ===")
print("Before BatchNorm: activations can have any mean and std")
print("After  BatchNorm: mean ≈ 0, std ≈ 1  (normalized per channel per batch)")
print("Why it helps: keeps gradient magnitudes stable during backpropagation")
print("EE analogy: like automatic gain control (AGC) in a receiver chain")

# =============================================
# VISUALIZE: what does ReLU actually do?
# =============================================
x_vals = np.linspace(-3, 3, 300)
relu   = np.maximum(0, x_vals)
leaky  = np.where(x_vals > 0, x_vals, 0.1 * x_vals)
sigmoid = 1 / (1 + np.exp(-x_vals))

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle('Activation Functions — The Non-Linearity That Makes Deep Learning Work',
             fontsize=12, fontweight='bold')

for ax, (name, y, note) in zip(axes, [
    ('ReLU\nmax(0, x)', relu, 'Most common in CNNs\nFast, avoids vanishing gradient'),
    ('Leaky ReLU\nmax(0.1x, x)', leaky, 'Used in YOLO and GANs\nFixes "dying ReLU" problem'),
    ('Sigmoid\n1/(1+e^-x)', sigmoid, 'Used for binary classification output\nSaturates → vanishing gradient in hidden layers'),
]):
    ax.plot(x_vals, y, linewidth=2.5, color='#1D9E75')
    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax.axvline(0, color='gray', linewidth=0.8, linestyle='--')
    ax.set_title(name, fontsize=11, fontweight='bold')
    ax.set_xlabel('Input value')
    ax.set_ylabel('Output value')
    ax.text(0.05, 0.95, note, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4))
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output/02_activation_functions.jpg', dpi=150)
plt.show()