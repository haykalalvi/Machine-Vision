"""
Day 12 - Script 4: Evaluation and Prediction Visualization
Load best model, run inference, visualize predictions vs ground truth.
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append('.')

from segmentation_dataset import (
    MVTecSegDatasetFixed, get_val_transform
)

DATA_ROOT = Path('/Users/alvi/OpenCV/data/leather')
IMG_SIZE  = 256
DEVICE    = torch.device(
    'mps'  if torch.backends.mps.is_available() else
    'cuda' if torch.cuda.is_available() else
    'cpu'
)

# Load model
model = smp.Unet(
    encoder_name='resnet34', encoder_weights=None,
    in_channels=3, classes=1, activation=None
)
model.load_state_dict(
    torch.load('output/best_unet.pt', map_location=DEVICE)
)
model = model.to(DEVICE)
model.eval()
print(f"Model loaded on {DEVICE}")

val_dataset = MVTecSegDatasetFixed(DATA_ROOT, 'val', transform=get_val_transform(IMG_SIZE))
val_loader  = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True)

def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    return torch.clamp(tensor * std + mean, 0, 1)

def compute_iou(pred, true, thresh=0.5):
    p = (pred > thresh).float(); t = true.float()
    inter = (p * t).sum(); union = p.sum() + t.sum() - inter
    return (inter / union).item() if union > 0 else 1.0

# ============================================================
# VISUALIZE PREDICTIONS
# ============================================================

n_samples = 8
samples   = []

with torch.no_grad():
    for imgs, masks in val_loader:
        imgs_dev = imgs.to(DEVICE)
        logits   = model(imgs_dev)
        preds    = torch.sigmoid(logits).cpu()
        iou      = compute_iou(preds[0], masks[0])
        samples.append({
            'img':  imgs[0], 'mask': masks[0,0],
            'pred': preds[0,0], 'iou': iou
        })
        if len(samples) >= n_samples:
            break

# Sort by IoU — show best and worst predictions
samples.sort(key=lambda x: x['iou'])

fig, axes = plt.subplots(n_samples, 4, figsize=(18, 5 * n_samples))
fig.suptitle(
    'U-Net Predictions on MVTec Leather\n'
    'Sorted: worst IoU (top) → best IoU (bottom)',
    fontsize=13, fontweight='bold'
)

col_titles = ['Input image', 'Ground truth mask', 'Predicted mask', 'Overlay comparison']
for col, title in enumerate(col_titles):
    axes[0][col].set_title(title, fontsize=11, fontweight='bold')

for row, s in enumerate(samples):
    img_d  = denormalize(s['img']).permute(1,2,0).numpy()
    mask_d = s['mask'].numpy()
    pred_d = s['pred'].numpy()
    pred_b = (pred_d > 0.5).astype(float)

    # Comparison overlay: Green=TP, Red=FN, Blue=FP
    comp = np.zeros((*mask_d.shape, 3))
    comp[(mask_d>0.5) & (pred_b>0.5)] = [0, 1, 0]   # TP = green
    comp[(mask_d>0.5) & (pred_b<0.5)] = [1, 0, 0]   # FN = red
    comp[(mask_d<0.5) & (pred_b>0.5)] = [0, 0, 1]   # FP = blue

    quality = 'Best' if row >= n_samples-2 else 'Worst' if row < 2 else ''
    color   = 'green' if row >= n_samples-2 else 'red' if row < 2 else 'black'

    axes[row][0].imshow(img_d)
    axes[row][0].set_ylabel(
        f'IoU={s["iou"]:.3f}\n{quality}',
        color=color, fontsize=10, fontweight='bold'
    )
    axes[row][0].axis('off')

    axes[row][1].imshow(mask_d, cmap='gray', vmin=0, vmax=1)
    axes[row][1].axis('off')

    axes[row][2].imshow(pred_d, cmap='hot', vmin=0, vmax=1)
    axes[row][2].set_title(
        'Pred (hot colormap\nbrighter=more confident)',
        fontsize=8
    ) if row == 0 else None
    axes[row][2].axis('off')

    axes[row][3].imshow(img_d)
    axes[row][3].imshow(comp, alpha=0.5)
    axes[row][3].set_title(
        'Green=TP | Red=FN | Blue=FP',
        fontsize=8
    ) if row == 0 else None
    axes[row][3].axis('off')

plt.tight_layout()
plt.savefig('output/04_predictions.jpg', dpi=120, bbox_inches='tight')
plt.show()
print("Saved: output/04_predictions.jpg")

# ============================================================
# COMPUTE OVERALL METRICS
# ============================================================

all_ious, all_dices = [], []
with torch.no_grad():
    for imgs, masks in val_loader:
        logits = model(imgs.to(DEVICE))
        preds  = torch.sigmoid(logits).cpu()
        p, t   = preds[0,0], masks[0,0]

        inter = ((p>0.5).float() * t).sum()
        union = (p>0.5).float().sum() + t.sum() - inter
        all_ious.append((inter/union).item() if union > 0 else 1.0)

        denom = (p>0.5).float().sum() + t.sum()
        all_dices.append((2*inter/denom).item() if denom > 0 else 1.0)

print("\n" + "="*50)
print("FINAL SEGMENTATION METRICS")
print("="*50)
print(f"Mean IoU:  {np.mean(all_ious):.4f}")
print(f"Mean Dice: {np.mean(all_dices):.4f}")
print(f"IoU std:   {np.std(all_ious):.4f}")
print()
print("Interpretation:")
print(f"  {'Excellent' if np.mean(all_ious) > 0.7 else 'Good' if np.mean(all_ious) > 0.5 else 'Needs improvement'}")
print("\n✓ Day 12 complete! Commit and move to Day 13.")