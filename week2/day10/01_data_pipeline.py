"""
Day 10 - Script 1: Data Pipeline
NEU Metal Surface Defect Dataset

Dataset path: /Users/alvi/OpenCV/data/NEU Metal Surface Defects Data/
Structure:
    train/ → 6 class folders
    valid/ → 6 class folders
    test/  → 6 class folders
"""

import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image

# ============================================================
# YOUR DATASET PATH — already set correctly for your machine
# ============================================================
DATA_ROOT = Path('/Users/alvi/OpenCV/data/NEU Metal Surface Defects Data')

# ============================================================
# SANITY CHECK — run this first to confirm paths are correct
# ============================================================
print("=" * 55)
print("SANITY CHECK — verifying dataset structure")
print("=" * 55)

for split in ['train', 'valid', 'test']:
    split_path = DATA_ROOT / split
    if not split_path.exists():
        print(f"  ERROR: {split_path} does not exist!")
        continue
    classes = sorted([f.name for f in split_path.iterdir() if f.is_dir()])
    n_imgs  = sum(len(list(f.glob('*.*'))) for f in split_path.iterdir() if f.is_dir())
    print(f"  {split}: {len(classes)} classes, {n_imgs} images")
    print(f"    Classes: {classes}")

print()

# ============================================================
# TRANSFORMS
# Training: with augmentation to simulate real camera variation
# Val/Test: no augmentation — deterministic only
# ============================================================

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    # Flip: defects appear on both sides of a metal strip
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),

    # Small rotation: simulate slight camera misalignment
    transforms.RandomRotation(degrees=15),

    # Color jitter: simulate different lighting on factory floor
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),

    # Random crop: simulate slight camera positioning variation
    transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),

    # MUST use ImageNet mean/std — pretrained model expects these exact values
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),   # deterministic, no randomness
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ============================================================
# LOAD DATASET — your data is already split into train/valid/test
# No need for random_split since folders already exist
# ============================================================

train_set = ImageFolder(root=str(DATA_ROOT / 'train'), transform=train_transform)
val_set   = ImageFolder(root=str(DATA_ROOT / 'valid'), transform=val_transform)
test_set  = ImageFolder(root=str(DATA_ROOT / 'test'),  transform=val_transform)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True,
                          num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_set,   batch_size=32, shuffle=False,
                          num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_set,  batch_size=32, shuffle=False,
                          num_workers=2, pin_memory=True)

CLASS_NAMES = train_set.classes
N_CLASSES   = len(CLASS_NAMES)

print("=" * 55)
print("DATASET LOADED SUCCESSFULLY")
print("=" * 55)
print(f"Classes ({N_CLASSES}): {CLASS_NAMES}")
print(f"Train : {len(train_set):>5} images | {len(train_loader):>3} batches")
print(f"Valid : {len(val_set):>5} images | {len(val_loader):>3} batches")
print(f"Test  : {len(test_set):>5} images | {len(test_loader):>3} batches")
print()

# Save dataset info for reuse in other scripts
torch.save({
    'class_names': CLASS_NAMES,
    'n_classes':   N_CLASSES,
}, 'output/dataset_info.pt')
print("Dataset info saved to output/dataset_info.pt")

# ============================================================
# VISUALIZE AUGMENTATION
# Show the same image with 9 different augmented versions
# This is what the model sees during training
# ============================================================

def denormalize(tensor):
    """Reverse ImageNet normalization for display"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return torch.clamp(tensor * std + mean, 0, 1)

# Pick one sample image from each class for display
sample_images = {}
for class_name in CLASS_NAMES:
    class_path = DATA_ROOT / 'train' / class_name
    imgs = sorted(class_path.glob('*.*'))
    if imgs:
        sample_images[class_name] = imgs[0]

# Show augmentation on first class
first_class = CLASS_NAMES[0]
sample_path = sample_images[first_class]
raw_img     = Image.open(sample_path).convert('RGB')

fig, axes = plt.subplots(2, 5, figsize=(20, 9))
fig.suptitle(
    f'Augmentation Pipeline — Class: {first_class}\n'
    'Same image, 9 different augmented versions — this is what your CNN sees each epoch',
    fontsize=13, fontweight='bold'
)

# Show original (no augmentation)
axes[0][0].imshow(raw_img)
axes[0][0].set_title('Original\n(no augmentation)', fontweight='bold', color='green')
axes[0][0].axis('off')

# Show 9 augmented versions
for i in range(1, 10):
    row, col  = divmod(i, 5)
    augmented = train_transform(raw_img)
    display   = denormalize(augmented).permute(1, 2, 0).numpy()
    axes[row][col].imshow(display)
    axes[row][col].set_title(f'Augmented v{i}', fontsize=9)
    axes[row][col].axis('off')

plt.tight_layout()
plt.savefig('output/01_augmentation_examples.jpg', dpi=150)
plt.show()
print("Saved: outputs/01_augmentation_examples.jpg")

# ============================================================
# VISUALIZE CLASS DISTRIBUTION
# ============================================================

class_counts = {}
for split in ['train', 'valid', 'test']:
    split_path = DATA_ROOT / split
    counts = {}
    for class_name in CLASS_NAMES:
        class_folder = split_path / class_name
        if class_folder.exists():
            counts[class_name] = len(list(class_folder.glob('*.*')))
        else:
            counts[class_name] = 0
    class_counts[split] = counts

colors = ['#1D9E75', '#378ADD', '#D85A30', '#7F77DD', '#BA7517', '#E05C8A']
x      = np.arange(N_CLASSES)
width  = 0.28

fig, ax = plt.subplots(figsize=(13, 6))
for i, (split, counts) in enumerate(class_counts.items()):
    vals = [counts[c] for c in CLASS_NAMES]
    bars = ax.bar(x + i * width, vals, width,
                  label=split.capitalize(),
                  color=colors[i], alpha=0.85)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                str(v), ha='center', fontsize=9, fontweight='bold')

ax.set_title(
    'NEU Metal Surface Defect — Images per Class per Split\n'
    'Balanced dataset → standard training without class weighting',
    fontsize=12, fontweight='bold'
)
ax.set_xticks(x + width)
ax.set_xticklabels(CLASS_NAMES, rotation=20, ha='right', fontsize=11)
ax.set_ylabel('Number of images')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('output/01_class_distribution.jpg', dpi=150)
plt.show()
print("Saved: outputs/01_class_distribution.jpg")

# ============================================================
# VISUALIZE SAMPLE IMAGES FROM EACH CLASS
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle(
    'NEU Metal Surface Defects — One Sample Per Class\n'
    'Study these carefully before training — know what you are detecting',
    fontsize=13, fontweight='bold'
)

for ax, (class_name, img_path) in zip(axes.flat, sample_images.items()):
    img = Image.open(img_path).convert('RGB')
    ax.imshow(img, cmap='gray')
    ax.set_title(class_name, fontsize=13, fontweight='bold')
    ax.axis('off')

plt.tight_layout()
plt.savefig('output/01_class_samples.jpg', dpi=150)
plt.show()
print("Saved: outputs/01_class_samples.jpg")

print("\n✓ Script 1 complete — data pipeline ready")
print(f"  Next: run 02_transfer_learning_strategies.py")