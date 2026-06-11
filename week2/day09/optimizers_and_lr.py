import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# =============================================
# PART 1: Gradient descent visualized
# Understand what the optimizer is actually doing
# =============================================

print("=" * 55)
print("UNDERSTANDING OPTIMIZERS")
print("=" * 55)

# A simple 1D loss landscape to visualize optimization
def loss_landscape(w):
    """A bumpy loss function with one global minimum"""
    return (w - 2.0)**2 + 0.5 * np.sin(5 * w) + 0.1 * np.cos(15 * w)

def loss_gradient(w):
    """Gradient of the loss (derivative)"""
    return 2 * (w - 2.0) + 2.5 * np.cos(5 * w) - 1.5 * np.sin(15 * w)

w_range = np.linspace(-1, 5, 500)
loss_vals = loss_landscape(w_range)

# Simulate optimization with different learning rates
def simulate_gd(start_w, lr, n_steps=40):
    w_history, loss_history = [start_w], [loss_landscape(start_w)]
    w = start_w
    for _ in range(n_steps):
        grad = loss_gradient(w)
        w    = w - lr * grad       # gradient descent update rule
        w_history.append(w)
        loss_history.append(loss_landscape(w))
    return w_history, loss_history

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Learning Rate Effect on Optimization\n'
             'Too small = too slow | Too large = diverges | Just right = converges',
             fontsize=12, fontweight='bold')

start_w = -0.5
lrs = [0.01, 0.15, 0.5]
titles = ['LR = 0.01\n(too small — crawls slowly)',
          'LR = 0.15\n(good — converges smoothly)', 
          'LR = 0.5\n(too large — bounces/diverges)']
colors = ['#d62728', '#1D9E75', '#ff7f0e']

for ax, lr, title, color in zip(axes, lrs, titles, colors):
    w_hist, loss_hist = simulate_gd(start_w, lr)
    
    ax.plot(w_range, loss_vals, 'k-', linewidth=2, label='Loss landscape', alpha=0.7)
    ax.plot(w_hist, [loss_landscape(w) for w in w_hist],
            'o-', color=color, linewidth=2, markersize=5, label='Optimizer path')
    ax.plot(w_hist[0], loss_landscape(w_hist[0]), 'g^', markersize=12, label='Start')
    ax.plot(w_hist[-1], loss_landscape(w_hist[-1]), 'rs', markersize=12, label='End')
    ax.set_title(title)
    ax.set_xlabel('Weight value (w)')
    ax.set_ylabel('Loss')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.5, 5)

plt.tight_layout()
plt.savefig('output/02a_learning_rate_effect.jpg', dpi=150)
plt.show()

# =============================================
# PART 2: SGD vs Adam — which to use and when
# =============================================

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(20, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.fc(x)

# Generate simple binary classification data
torch.manual_seed(42)
X = torch.randn(500, 20)
y = (X[:, 0] + X[:, 1] > 0).float().unsqueeze(1)

dataset    = torch.utils.data.TensorDataset(X, y)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

optimizers_config = {
    'SGD (lr=0.01)':          lambda p: torch.optim.SGD(p, lr=0.01),
    'SGD + Momentum (0.9)':   lambda p: torch.optim.SGD(p, lr=0.01, momentum=0.9),
    'Adam (lr=1e-3)':         lambda p: torch.optim.Adam(p, lr=1e-3),
    'AdamW (lr=1e-3)':        lambda p: torch.optim.AdamW(p, lr=1e-3, weight_decay=1e-2),
}

criterion  = nn.BCEWithLogitsLoss()
all_losses = {}

for opt_name, opt_fn in optimizers_config.items():
    model_cmp = SimpleNet()
    optimizer = opt_fn(model_cmp.parameters())
    epoch_losses = []
    
    for epoch in range(30):
        epoch_loss = 0
        for xb, yb in dataloader:
            optimizer.zero_grad()
            loss = criterion(model_cmp(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_losses.append(epoch_loss / len(dataloader))
    
    all_losses[opt_name] = epoch_losses
    print(f"{opt_name:30s} → Final loss: {epoch_losses[-1]:.4f}")

fig, ax = plt.subplots(figsize=(12, 6))
colors = ['#d62728', '#ff7f0e', '#1D9E75', '#9467bd']
for (name, losses), color in zip(all_losses.items(), colors):
    ax.plot(losses, label=name, linewidth=2.5, color=color)

ax.set_xlabel('Epoch')
ax.set_ylabel('Training Loss')
ax.set_title('Optimizer Comparison: SGD vs Adam vs AdamW\n'
             'For CNN training: start with AdamW, switch to SGD+momentum for fine-tuning')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('output/02b_optimizer_comparison.jpg', dpi=150)
plt.show()

print("\n=== When to use which optimizer ===")
print("AdamW    → Default choice for new projects. Fast convergence, handles LR well.")
print("SGD+mom  → Better final accuracy when fine-tuning pretrained models.")
print("Adam     → Same as AdamW but without weight decay. Use AdamW instead.")
print("\nFor your manufacturing QC project: start with AdamW(lr=1e-3, weight_decay=1e-2)")