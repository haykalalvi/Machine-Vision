"""
Day 12 - Script 3: U-Net Training on MVTec AD Leather
Uses segmentation-models-pytorch for the model.
Uses combined Dice + BCE loss for handling pixel imbalance.
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import time

os.environ["WANDB_MODE"] = "offline"
import wandb

# ============================================================
# RE-USE DATASET FROM SCRIPT 2
# ============================================================
# Fixed Import: Make sure you renamed the previous file to 'segmentation_dataset.py'
from segmentation_dataset import (
    MVTecSegDataset, get_train_transform, get_val_transform
)

# ============================================================
# LOSS FUNCTIONS & METRICS
# ============================================================

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs   = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / \
               (probs.sum() + targets.sum() + self.smooth)
        return 1.0 - dice


class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super().__init__()
        self.dice     = DiceLoss()
        self.bce      = nn.BCEWithLogitsLoss()
        self.dice_w   = dice_weight
        self.bce_w    = bce_weight

    def forward(self, logits, targets):
        return self.dice_w * self.dice(logits, targets) + \
               self.bce_w  * self.bce(logits, targets)


def compute_iou(pred_mask, true_mask, threshold=0.5):
    pred_bin = (pred_mask > threshold).float()
    true_bin = true_mask.float()

    intersection = (pred_bin * true_bin).sum()
    union        = pred_bin.sum() + true_bin.sum() - intersection
    if union == 0:
        return 1.0 
    return (intersection / union).item()


def compute_dice(pred_mask, true_mask, threshold=0.5):
    pred_bin     = (pred_mask > threshold).float()
    true_bin     = true_mask.float()
    intersection = (pred_bin * true_bin).sum()
    denom        = pred_bin.sum() + true_bin.sum()
    if denom == 0:
        return 1.0
    return (2.0 * intersection / denom).item()

# ============================================================
# TRAINING FUNCTIONS
# ============================================================

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_iou, total_dice = 0, 0, 0

    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()

        logits = model(imgs)
        loss   = criterion(logits, masks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        with torch.no_grad():
            preds = torch.sigmoid(logits)
            total_loss += loss.item()
            total_iou  += compute_iou(preds, masks)
            total_dice += compute_dice(preds, masks)

    n = len(loader)
    return total_loss/n, total_iou/n, total_dice/n


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, total_iou, total_dice = 0, 0, 0

    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)
        loss   = criterion(logits, masks)
        preds  = torch.sigmoid(logits)
        total_loss += loss.item()
        total_iou  += compute_iou(preds, masks)
        total_dice += compute_dice(preds, masks)

    n = len(loader)
    return total_loss/n, total_iou/n, total_dice/n


# ============================================================
# MAIN EXECUTION BLOCK (Protects Multiprocessing)
# ============================================================
if __name__ == '__main__':
    
    DATA_ROOT = Path('/Users/alvi/OpenCV/data/leather')
    IMG_SIZE  = 256
    DEVICE    = torch.device(
        'mps'  if torch.backends.mps.is_available() else
        'cuda' if torch.cuda.is_available() else
        'cpu'
    )
    print(f"Device: {DEVICE}")

    # 1. Datasets and Loaders
    train_dataset = MVTecSegDataset(DATA_ROOT, 'train', transform=get_train_transform(IMG_SIZE))
    val_dataset   = MVTecSegDataset(DATA_ROOT, 'val',   transform=get_val_transform(IMG_SIZE))

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=8, shuffle=False, num_workers=2, pin_memory=True)

    # 2. Model Init
    model = smp.Unet(
        encoder_name='resnet34',          
        encoder_weights='imagenet',       
        in_channels=3,
        classes=1,
        activation=None,                  
    )
    model = model.to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"U-Net parameters: {total_params:,}")

    # 3. Training Config
    EPOCHS   = 30
    BASE_LR  = 1e-3

    criterion = CombinedLoss(dice_weight=0.5, bce_weight=0.5)
    optimizer = torch.optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': BASE_LR * 0.1},
        {'params': model.decoder.parameters(), 'lr': BASE_LR},
        {'params': model.segmentation_head.parameters(), 'lr': BASE_LR},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-5
    )

    wandb.init(
        project="day12-unet-mvtec",
        name="unet-resnet34-leather",
        config={
            'model':    'UNet-ResNet34',
            'dataset':  'MVTec-leather',
            'epochs':   EPOCHS,
            'img_size': IMG_SIZE,
            'loss':     'Dice+BCE',
        }
    )

    # 4. Training Loop
    print(f"\nStarting training for {EPOCHS} epochs...")
    print(f"{'Epoch':>6} | {'Tr Loss':>8} | {'Tr IoU':>7} | {'Val Loss':>9} | {'Val IoU':>8} | {'Val Dice':>9}")
    print("-" * 65)

    history     = {'tr_loss':[], 'tr_iou':[], 'val_loss':[], 'val_iou':[], 'val_dice':[]}
    best_val_iou = 0.0
    t_start      = time.time()
    
    Path('output').mkdir(exist_ok=True) # Ensure output directory exists

    for epoch in range(EPOCHS):
        tr_loss, tr_iou, tr_dice    = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_iou, val_dice = validate(model, val_loader, criterion, DEVICE)
        scheduler.step()

        history['tr_loss'].append(tr_loss)
        history['tr_iou'].append(tr_iou)
        history['val_loss'].append(val_loss)
        history['val_iou'].append(val_iou)
        history['val_dice'].append(val_dice)

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), 'output/best_unet.pt')
            best_flag = ' ← best'
        else:
            best_flag = ''

        wandb.log({
            'train/loss': tr_loss, 'train/iou': tr_iou,
            'val/loss': val_loss,  'val/iou': val_iou,
            'val/dice': val_dice,  'epoch': epoch + 1,
        })

        print(f"{epoch+1:6d} | {tr_loss:8.4f} | {tr_iou:7.4f} | "
              f"{val_loss:9.4f} | {val_iou:8.4f} | {val_dice:9.4f}{best_flag}")

    elapsed = time.time() - t_start
    print(f"\nTraining complete in {elapsed/60:.1f} minutes")
    print(f"Best validation IoU: {best_val_iou:.4f}")
    wandb.finish()

    # 5. Training Curves
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f'U-Net Training on MVTec Leather | Best Val IoU: {best_val_iou:.4f}',
                 fontsize=13, fontweight='bold')

    epochs_range = range(1, EPOCHS + 1)

    axes[0].plot(epochs_range, history['tr_loss'],  label='Train', color='#D85A30', linewidth=2)
    axes[0].plot(epochs_range, history['val_loss'],  label='Val',   color='#378ADD', linewidth=2)
    axes[0].set_title('Loss (Dice + BCE)'); axes[0].set_xlabel('Epoch')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs_range, history['tr_iou'],   label='Train IoU', color='#D85A30', linewidth=2)
    axes[1].plot(epochs_range, history['val_iou'],   label='Val IoU',   color='#378ADD', linewidth=2)
    axes[1].set_title('IoU Score'); axes[1].set_xlabel('Epoch')
    axes[1].axhline(0.5, color='gray', linestyle='--', label='0.5 baseline')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs_range, history['val_dice'],  label='Val Dice', color='#1D9E75', linewidth=2)
    axes[2].set_title('Validation Dice Score'); axes[2].set_xlabel('Epoch')
    axes[2].legend(); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output/03_training_curves.jpg', dpi=150)
    plt.show()
    print("Saved: output/03_training_curves.jpg")