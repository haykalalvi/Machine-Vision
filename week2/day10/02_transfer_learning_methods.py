"""
Day 10 - Script 2: Transfer Learning Strategies Comparison
Trains 3 versions of EfficientNet-B0 on NEU Metal Surface Defects:
  1. Feature Extraction  — freeze all, train classifier head only
  2. Partial Fine-tune   — unfreeze last 3 blocks + classifier
  3. Full Fine-tune      — unfreeze everything, layerwise LR

Dataset path: /Users/alvi/OpenCV/data/NEU Metal Surface Defects Data/
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, classification_report
import time
import os

# ============================================================
# W&B SETUP — set to offline if you had trouble before
# Change to "online" once your W&B login is working
# ============================================================
os.environ["WANDB_MODE"] = "online"   # change to "online" when ready
import wandb

# ============================================================
# YOUR DATASET PATH
# ============================================================
DATA_ROOT = Path('/Users/alvi/OpenCV/data/NEU Metal Surface Defects Data')

# ============================================================
# TRANSFORMS — same as Script 1
# ============================================================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# ============================================================
# MODEL BUILDERS — 3 different freeze strategies
# All use EfficientNet-B0 pretrained on ImageNet
# ============================================================

def build_feature_extractor(n_classes):
    """
    Strategy 1: Freeze ALL pretrained layers.
    Only the new classifier head is trained.
    Fastest. Use when: very small dataset OR very similar to ImageNet.
    """
    model = models.efficientnet_b0(weights='IMAGENET1K_V1')

    # Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Replace classifier — only this part trains
    in_features  = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, n_classes)
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  [Feature Extractor] Trainable: {trainable:,} / {total:,} ({trainable/total:.1%})")
    return model


def build_partial_finetune(n_classes):
    """
    Strategy 2: Freeze early layers, unfreeze last 3 blocks + classifier.
    Best balance for medium-sized datasets.
    Most common approach in production.
    """
    model = models.efficientnet_b0(weights='IMAGENET1K_V1')

    # Freeze all first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last 3 feature blocks (6, 7, 8)
    for i in [6, 7, 8]:
        for param in model.features[i].parameters():
            param.requires_grad = True

    # Replace and unfreeze classifier
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, n_classes)
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  [Partial Fine-tune] Trainable: {trainable:,} / {total:,} ({trainable/total:.1%})")
    return model


def build_full_finetune(n_classes):
    """
    Strategy 3: All layers trainable.
    Highest potential accuracy.
    Use layerwise LR to prevent catastrophic forgetting.
    """
    model = models.efficientnet_b0(weights='IMAGENET1K_V1')

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, n_classes)
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  [Full Fine-tune]    Trainable: {trainable:,} / {total:,} ({trainable/total:.1%})")
    return model


def get_optimizer(model, strategy, base_lr=1e-3):
    """
    Layerwise learning rates:
    Pretrained layers  → small LR (preserve learned features)
    New classifier     → large LR (must learn from scratch)
    """
    if strategy == 'feature_extractor':
        return torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=base_lr, weight_decay=1e-2
        )
    elif strategy == 'partial_finetune':
        return torch.optim.AdamW([
            {'params': model.features[6].parameters(), 'lr': base_lr * 0.1},
            {'params': model.features[7].parameters(), 'lr': base_lr * 0.1},
            {'params': model.features[8].parameters(), 'lr': base_lr * 0.5},
            {'params': model.classifier.parameters(),  'lr': base_lr},
        ], weight_decay=1e-2)
    else:  # full_finetune
        return torch.optim.AdamW([
            # Early layers: tiny LR
            {'params': model.features[:5].parameters(), 'lr': base_lr * 0.01},
            # Later layers: medium LR
            {'params': model.features[5:].parameters(), 'lr': base_lr * 0.1},
            # Classifier: full LR
            {'params': model.classifier.parameters(),   'lr': base_lr},
        ], weight_decay=1e-2)


# ============================================================
# TRAINING & EVALUATION FUNCTIONS
# ============================================================

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        correct    += (out.argmax(1) == labels).sum().item()
        total      += labels.size(0)
    return total_loss / len(loader), correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs   = imgs.to(device)
        preds  = model(imgs).argmax(1).cpu()
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc = (all_preds == all_labels).mean()
    f1  = f1_score(all_labels, all_preds, average='macro')
    return acc, f1, all_preds, all_labels


# ============================================================
# MAIN EXECUTION BLOCK (Fixes multiprocessing error)
# ============================================================
if __name__ == '__main__':
    
    # Properly detect Mac GPUs (MPS) for faster training
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')
    elif torch.backends.mps.is_available():
        DEVICE = torch.device('mps')
    else:
        DEVICE = torch.device('cpu')

    # ============================================================
    # LOAD DATASET
    # ============================================================
    train_set = ImageFolder(root=str(DATA_ROOT / 'train'), transform=train_transform)
    val_set   = ImageFolder(root=str(DATA_ROOT / 'valid'), transform=val_transform)
    test_set  = ImageFolder(root=str(DATA_ROOT / 'test'),  transform=val_transform)

    BATCH = 32
    train_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH, shuffle=False,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=BATCH, shuffle=False,
                              num_workers=2, pin_memory=True)

    CLASS_NAMES = train_set.classes
    N_CLASSES   = len(CLASS_NAMES)
    EPOCHS      = 10

    print(f"Device    : {DEVICE}")
    print(f"Classes   : {N_CLASSES} → {CLASS_NAMES}")
    print(f"Train     : {len(train_set)} images | {len(train_loader)} batches")
    print(f"Valid     : {len(val_set)} images  | {len(val_loader)} batches")
    print(f"Test      : {len(test_set)} images  | {len(test_loader)} batches")
    print()

    # Ensure output directory exists
    Path("output").mkdir(parents=True, exist_ok=True)

    # ============================================================
    # TRAIN ALL THREE STRATEGIES
    # ============================================================

    strategies = {
        'feature_extractor': build_feature_extractor,
        'partial_finetune':  build_partial_finetune,
        'full_finetune':     build_full_finetune,
    }

    criterion   = nn.CrossEntropyLoss()
    all_results = {}

    for strategy_name, build_fn in strategies.items():
        print(f"\n{'='*55}")
        print(f"Strategy: {strategy_name.upper()}")
        print('='*55)

        wandb.init(
            project="day10-neu-transfer-learning",
            name=f"efficientnet-b0-{strategy_name}",
            config={
                'strategy':  strategy_name,
                'epochs':    EPOCHS,
                'backbone':  'efficientnet_b0',
                'dataset':   'NEU Metal Surface Defects',
                'n_classes': N_CLASSES,
            }
        )

        model     = build_fn(N_CLASSES).to(DEVICE)
        optimizer = get_optimizer(model, strategy_name)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS
        )

        history = {
            'train_loss': [], 'train_acc': [],
            'val_acc':    [], 'val_f1':    []
        }
        best_val_acc = 0.0
        t_start      = time.time()

        for epoch in range(EPOCHS):
            tr_loss, tr_acc   = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
            val_acc, val_f1, _, _ = evaluate(model, val_loader, DEVICE)
            scheduler.step()

            history['train_loss'].append(tr_loss)
            history['train_acc'].append(tr_acc)
            history['val_acc'].append(val_acc)
            history['val_f1'].append(val_f1)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    model.state_dict(),
                    f'output/best_{strategy_name}.pt'
                )

            wandb.log({
                'train/loss': tr_loss,
                'train/acc':  tr_acc,
                'val/acc':    val_acc,
                'val/f1':     val_f1,
                'epoch':      epoch + 1,
            })

            best_flag = ' ← best' if val_acc == best_val_acc else ''
            print(
                f"  Epoch {epoch+1:2d}/{EPOCHS} | "
                f"Loss: {tr_loss:.4f} | "
                f"Train: {tr_acc:.3f} | "
                f"Val: {val_acc:.3f} | "
                f"F1: {val_f1:.3f}"
                f"{best_flag}"
            )

        elapsed = time.time() - t_start

        # Final test evaluation using best checkpoint
        model.load_state_dict(
            torch.load(f'output/best_{strategy_name}.pt', map_location=DEVICE)
        )
        test_acc, test_f1, test_preds, test_labels_arr = evaluate(model, test_loader, DEVICE)

        all_results[strategy_name] = {
            'history':   history,
            'test_acc':  test_acc,
            'test_f1':   test_f1,
            'time_min':  elapsed / 60,
            'preds':     test_preds,
            'labels':    test_labels_arr,
        }

        print(f"\n  TEST → Acc: {test_acc:.4f} | F1: {test_f1:.4f} | Time: {elapsed/60:.1f} min")
        wandb.finish()

    # ============================================================
    # COMPREHENSIVE COMPARISON VISUALIZATION
    # ============================================================

    labels_map = {
        'feature_extractor': 'Feature Extraction',
        'partial_finetune':  'Partial Fine-tune',
        'full_finetune':     'Full Fine-tune',
    }
    colors_map = {
        'feature_extractor': '#D85A30',
        'partial_finetune':  '#378ADD',
        'full_finetune':     '#1D9E75',
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(
        'Transfer Learning Strategy Comparison\n'
        'EfficientNet-B0 on NEU Metal Surface Defects (6 classes)',
        fontsize=14, fontweight='bold'
    )

    # --- Plot 1: Validation accuracy curves ---
    ax = axes[0][0]
    for name, res in all_results.items():
        ax.plot(
            range(1, EPOCHS + 1),
            res['history']['val_acc'],
            label=labels_map[name],
            color=colors_map[name],
            linewidth=2.5, marker='o', markersize=4
        )
    ax.set_title('Validation Accuracy Over Epochs')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Plot 2: Training loss curves ---
    ax = axes[0][1]
    for name, res in all_results.items():
        ax.plot(
            range(1, EPOCHS + 1),
            res['history']['train_loss'],
            label=labels_map[name],
            color=colors_map[name],
            linewidth=2.5, marker='o', markersize=4
        )
    ax.set_title('Training Loss Over Epochs')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Plot 3: Final test comparison bar chart ---
    ax    = axes[0][2]
    names = list(all_results.keys())
    accs  = [all_results[n]['test_acc'] * 100 for n in names]
    f1s   = [all_results[n]['test_f1']  * 100 for n in names]
    x_pos = np.arange(len(names))
    w     = 0.35
    bars1 = ax.bar(x_pos - w/2, accs, w,
                   label='Accuracy (%)',
                   color=[colors_map[n] for n in names])
    bars2 = ax.bar(x_pos + w/2, f1s,  w,
                   label='Macro F1 (%)',
                   color=[colors_map[n] for n in names], alpha=0.6)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([labels_map[n] for n in names], fontsize=9)
    ax.set_title('Final Test Performance')
    ax.set_ylabel('Score (%)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars1, accs):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', fontsize=9, fontweight='bold')

    # --- Plot 4: Training time ---
    ax    = axes[1][0]
    times = [all_results[n]['time_min'] for n in names]
    bars  = ax.bar(
        [labels_map[n] for n in names],
        times,
        color=[colors_map[n] for n in names]
    )
    ax.set_title('Training Time (minutes)')
    ax.set_ylabel('Minutes')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.1,
                f'{t:.1f}m', ha='center', fontweight='bold')

    # --- Plot 5: Per-class F1 for best strategy ---
    best_strategy = max(all_results, key=lambda k: all_results[k]['test_f1'])
    best_res      = all_results[best_strategy]
    ax            = axes[1][1]
    report        = classification_report(
        best_res['labels'], best_res['preds'],
        target_names=CLASS_NAMES, output_dict=True
    )
    per_class_f1 = [report[c]['f1-score'] for c in CLASS_NAMES]
    bar_colors   = ['#1D9E75' if f >= 0.9 else '#D85A30' for f in per_class_f1]
    ax.barh(CLASS_NAMES, per_class_f1, color=bar_colors, alpha=0.85)
    ax.set_title(
        f'Per-Class F1 ({labels_map[best_strategy]})\n'
        f'Green = F1 ≥ 0.90 | Red = F1 < 0.90'
    )
    ax.set_xlabel('F1 Score')
    ax.axvline(0.9, color='black', linestyle='--', linewidth=1.5, label='Target 0.90')
    ax.set_xlim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    for i, v in enumerate(per_class_f1):
        ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=10)

    # --- Plot 6: Summary table ---
    ax = axes[1][2]
    ax.axis('off')
    table_data = [['Strategy', 'Test Acc', 'Test F1', 'Time']]
    for name in names:
        r = all_results[name]
        table_data.append([
            labels_map[name],
            f"{r['test_acc']:.3f}",
            f"{r['test_f1']:.3f}",
            f"{r['time_min']:.1f}m"
        ])
    table = ax.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        loc='center', cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.4)
    ax.set_title('Strategy Comparison Summary', fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('output/02_strategy_comparison.jpg', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: output/02_strategy_comparison.jpg")

    # ============================================================
    # PRINT FINAL RECOMMENDATION
    # ============================================================
    best_name = max(all_results, key=lambda k: all_results[k]['test_f1'])
    print("\n" + "="*55)
    print("RESULTS SUMMARY")
    print("="*55)
    for name in names:
        r = all_results[name]
        print(f"  {labels_map[name]:22s} → "
              f"Acc: {r['test_acc']:.3f} | "
              f"F1: {r['test_f1']:.3f} | "
              f"Time: {r['time_min']:.1f}min")

    print(f"\nBest strategy: {labels_map[best_name]}")
    print(f"Best model saved to: output/best_{best_name}.pt")
    print("\n✓ Script 2 complete — run 03_gradcam_visualization.py next")