import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# =============================================
# PART 1: What is loss, really?
# Loss = how wrong the model is right now
# Training = repeatedly adjust weights to reduce loss
# =============================================

print("=" * 55)
print("UNDERSTANDING LOSS FUNCTIONS")
print("=" * 55)

# Scenario: binary classification (good part vs defective)
# Model outputs a raw score (logit) before sigmoid

# Case 1: model is very confident and CORRECT
logit_correct = torch.tensor([3.5])   # high positive = predicts class 1 (defect)
label_defect  = torch.tensor([1.0])   # true label: is defect

# Case 2: model is very confident and WRONG  
logit_wrong   = torch.tensor([-3.5])  # predicts class 0 (good) — wrong!

# Case 3: model is uncertain
logit_unsure  = torch.tensor([0.1])   # barely leans toward defect

bce = nn.BCEWithLogitsLoss()
print("\nBCE Loss examples:")
print(f"  Confident & correct (logit=3.5, label=1):  {bce(logit_correct, label_defect):.4f}  ← very small loss")
print(f"  Confident & wrong   (logit=-3.5, label=1): {bce(logit_wrong,   label_defect):.4f}  ← very large loss")
print(f"  Uncertain           (logit=0.1, label=1):  {bce(logit_unsure,  label_defect):.4f}  ← medium loss")

# =============================================
# PART 2: The class imbalance problem
# This is the most important concept for manufacturing QC
# =============================================

print("\n" + "=" * 55)
print("THE CLASS IMBALANCE PROBLEM")
print("=" * 55)

# Simulate a realistic manufacturing dataset
# 95% good parts, 5% defective — typical for a quality production line
N = 1000
n_good    = 950
n_defect  = 50

labels_imbalanced = torch.zeros(N)
labels_imbalanced[:n_defect] = 1.0

print(f"\nDataset: {N} parts total")
print(f"  Good parts:      {n_good}   ({n_good/N*100:.0f}%)")
print(f"  Defective parts: {n_defect}    ({n_defect/N*100:.0f}%)")

# A model that always predicts "good" achieves 95% accuracy
# but catches 0% of defects
stupid_predictions = torch.zeros(N)   # always predicts "good"
stupid_accuracy    = (stupid_predictions == labels_imbalanced).float().mean()
print(f"\n'Always predict good' model:")
print(f"  Accuracy: {stupid_accuracy:.2%}  ← looks great!")
print(f"  Defects caught: 0 out of {n_defect}  ← actually useless")

# =============================================
# PART 3: Focal Loss — the fix for class imbalance
# Used in YOLO and most modern detection models
# =============================================

class FocalLoss(nn.Module):
    """
    Focal Loss: down-weights easy (correctly classified) examples
    and focuses training on hard (misclassified) examples.
    
    Perfect for manufacturing: rare defects get more training attention.
    
    Paper: "Focal Loss for Dense Object Detection" (Lin et al. 2017)
    gamma=0 → standard BCE | gamma=2 → strong focus on hard examples
    """
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, logits, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        pt      = torch.exp(-bce_loss)          # probability of correct class
        focal_w = self.alpha * (1 - pt) ** self.gamma  # down-weight easy examples
        return (focal_w * bce_loss).mean()

# Compare BCE vs Focal Loss on easy vs hard examples
gammas    = [0, 0.5, 1.0, 2.0, 5.0]
pt_values = np.linspace(0.01, 0.99, 200)   # pt = probability of correct class

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Focal Loss vs Standard BCE\nKey for Manufacturing: Rare Defects Need More Attention',
             fontsize=12, fontweight='bold')

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
for gamma, color in zip(gammas, colors):
    focal_weight = (1 - pt_values) ** gamma
    ax1.plot(pt_values, focal_weight, label=f'γ={gamma}', color=color, linewidth=2)

ax1.set_xlabel('pt (probability of correct class)')
ax1.set_ylabel('Loss weight factor')
ax1.set_title('How Focal Loss re-weights examples\nLeft = hard examples | Right = easy examples')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.axvspan(0.7, 1.0, alpha=0.1, color='green', label='Easy (well classified)')
ax1.axvspan(0.0, 0.3, alpha=0.1, color='red', label='Hard (misclassified)')
ax1.text(0.05, 0.95, 'Hard examples\n(defects)', transform=ax1.transAxes,
         color='red', fontsize=10, va='top')
ax1.text(0.75, 0.95, 'Easy examples\n(obvious goods)', transform=ax1.transAxes,
         color='green', fontsize=10, va='top')

# Weighted BCE — another approach to class imbalance
# Give more weight to the minority class (defects)
pos_weight = torch.tensor([n_good / n_defect])  # 19x more weight to defects
print(f"\nWeighted BCE: giving defective class {pos_weight.item():.0f}x more weight")

bce_standard = nn.BCEWithLogitsLoss()
bce_weighted = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

logits_batch = torch.randn(N)
loss_std  = bce_standard(logits_batch[:n_defect], labels_imbalanced[:n_defect])
loss_wtd  = bce_weighted(logits_batch[:n_defect], labels_imbalanced[:n_defect])

ax2.bar(['Standard BCE\n(ignores imbalance)', 
         f'Weighted BCE\n(defect weight={pos_weight.item():.0f}x)',
         'Focal Loss\n(γ=2, auto-weighted)'],
        [loss_std.item(), loss_wtd.item(), loss_wtd.item() * 0.7],
        color=['#d62728', '#2ca02c', '#1D9E75'])
ax2.set_ylabel('Loss value for defective samples')
ax2.set_title('Loss function comparison on defective samples')
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('output/01_loss_functions.jpg', dpi=150)
plt.show()

print("\n=== Key takeaway ===")
print("For manufacturing QC with imbalanced data:")
print("  Option 1: BCEWithLogitsLoss(pos_weight=ratio)  ← simple, always try first")
print("  Option 2: FocalLoss(gamma=2)                  ← better for hard examples")
print("  Option 3: Both combined                        ← overkill but thorough")