"""
Day 12 - Script 1: MVTec AD Dataset Exploration
Understand the structure before building the segmentation model.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_ROOT = Path('/Users/alvi/OpenCV/data/leather')

# ============================================================
# EXPLORE DATASET STRUCTURE
# ============================================================
print("=" * 55)
print("MVTec AD — Leather Category Structure")
print("=" * 55)

# Count training good images
train_good = list((DATA_ROOT / 'train' / 'good').glob('*.png'))
print(f"\nTraining images (good only): {len(train_good)}")
print("Note: We ONLY train on good images — no defect labels needed for Day 13")

# Count test images per defect type
test_path    = DATA_ROOT / 'test'
defect_types = sorted([d.name for d in test_path.iterdir() if d.is_dir()])
print(f"\nTest defect types: {defect_types}")

total_defect = 0
for dtype in defect_types:
    imgs = list((test_path / dtype).glob('*.png'))
    print(f"  {dtype:15s}: {len(imgs):3d} images")
    if dtype != 'good':
        total_defect += len(imgs)

print(f"\nTotal defective test images: {total_defect}")
print(f"Total good test images: {len(list((test_path/'good').glob('*.png')))}")

# ============================================================
# VISUALIZE: image + ground truth mask pairs
# ============================================================
gt_path = DATA_ROOT / 'ground_truth'

fig, axes = plt.subplots(len(defect_types)-1, 4, figsize=(18, 5*(len(defect_types)-1)))
fig.suptitle(
    'MVTec AD Leather — Defect Images with Ground Truth Masks\n'
    'Day 12: we train U-Net to predict these masks | Day 13: no masks needed',
    fontsize=13, fontweight='bold'
)

col_titles = ['Original image', 'Ground truth mask', 'Mask overlay', 'Defect statistics']
for col, title in enumerate(col_titles):
    axes[0][col].set_title(title, fontsize=11, fontweight='bold')

row = 0
for dtype in defect_types:
    if dtype == 'good':
        continue

    # Get first defective image and its mask
    defect_imgs = sorted((test_path / dtype).glob('*.png'))
    if not defect_imgs:
        continue

    img_path  = defect_imgs[0]
    img       = cv2.imread(str(img_path))
    img_rgb   = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Load corresponding ground truth mask
    mask_path = gt_path / dtype / (img_path.stem + '_mask.png')
    if mask_path.exists():
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    else:
        mask = np.zeros(img.shape[:2], dtype=np.uint8)

    # Compute defect statistics
    defect_pixels = (mask > 127).sum()
    total_pixels  = mask.size
    defect_pct    = defect_pixels / total_pixels * 100

    # Create overlay
    overlay     = img_rgb.copy()
    defect_mask = mask > 127
    overlay[defect_mask] = [255, 50, 50]  # red overlay on defect

    # Plot
    axes[row][0].imshow(img_rgb)
    axes[row][0].set_ylabel(dtype, fontsize=11, fontweight='bold')
    axes[row][0].axis('off')

    axes[row][1].imshow(mask, cmap='gray')
    axes[row][1].axis('off')

    axes[row][2].imshow(cv2.addWeighted(img_rgb, 0.7,
                        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR), 0.3, 0)
                        if False else overlay)
    blended = (0.6 * img_rgb + 0.4 * np.stack([mask>127]*3, axis=-1) * np.array([255,0,0])).astype(np.uint8)
    axes[row][2].imshow(blended)
    axes[row][2].axis('off')

    # Stats text
    axes[row][3].axis('off')
    stats_text = (
        f"Defect type: {dtype}\n\n"
        f"Image size: {img.shape[1]}×{img.shape[0]}px\n\n"
        f"Defect pixels: {defect_pixels:,}\n"
        f"Total pixels: {total_pixels:,}\n"
        f"Defect coverage: {defect_pct:.2f}%\n\n"
        f"Challenge:\n"
        f"{'Small defect' if defect_pct < 2 else 'Medium defect' if defect_pct < 10 else 'Large defect'}\n"
        f"({'Hard to detect' if defect_pct < 2 else 'Moderate' if defect_pct < 10 else 'Easier to detect'})"
    )
    axes[row][3].text(0.1, 0.5, stats_text, transform=axes[row][3].transAxes,
                      fontsize=11, va='center', family='monospace',
                      bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    row += 1

plt.tight_layout()
plt.savefig('output/01_dataset_exploration.jpg', dpi=150, bbox_inches='tight')
plt.show()
print("\nSaved: outputs/01_dataset_exploration.jpg")

# ============================================================
# CLASS IMBALANCE ANALYSIS — critical for segmentation
# ============================================================
print("\n" + "="*55)
print("CLASS IMBALANCE ANALYSIS")
print("="*55)
print("In segmentation, class imbalance = pixel imbalance")
print("Defect pixels are RARE compared to background pixels")
print("This is why we use Dice Loss instead of CrossEntropy\n")

all_defect_pcts = []
for dtype in defect_types:
    if dtype == 'good':
        continue
    for img_path in sorted((test_path / dtype).glob('*.png'))[:5]:
        mask_path = gt_path / dtype / (img_path.stem + '_mask.png')
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            pct  = (mask > 127).sum() / mask.size * 100
            all_defect_pcts.append(pct)

print(f"Average defect coverage across all samples: {np.mean(all_defect_pcts):.2f}%")
print(f"This means background:defect pixel ratio ≈ {100/np.mean(all_defect_pcts):.0f}:1")
print("→ CrossEntropy would ignore defect pixels (too few)")
print("→ Dice Loss directly optimizes overlap — correct choice")