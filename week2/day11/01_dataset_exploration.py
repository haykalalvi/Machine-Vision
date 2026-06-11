import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import yaml
import random

DATA_ROOT = Path('/Users/alvi/OpenCV/data/PCB Defect')

# Load dataset config
with open(DATA_ROOT / 'data.yaml') as f:
    config = yaml.safe_load(f)

CLASS_NAMES = config['names']
N_CLASSES   = len(CLASS_NAMES)
print(f"Classes ({N_CLASSES}): {CLASS_NAMES}")

# Count images and labels per split
for split in ['train', 'valid', 'test']:
    split_path = DATA_ROOT / split
    if not split_path.exists():
        continue
    n_imgs   = len(list((split_path/'images').glob('*')))
    n_labels = len(list((split_path/'labels').glob('*.txt')))
    print(f"  {split}: {n_imgs} images, {n_labels} label files")

# =============================================
# VISUALIZE GROUND TRUTH ANNOTATIONS
# =============================================

def load_yolo_boxes(label_path, img_w, img_h):
    """Convert YOLO normalized format to pixel coordinates"""
    boxes = []
    if not label_path.exists():
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id    = int(parts[0])
            x_center  = float(parts[1]) * img_w
            y_center  = float(parts[2]) * img_h
            width     = float(parts[3]) * img_w
            height    = float(parts[4]) * img_h
            x1 = x_center - width/2
            y1 = y_center - height/2
            boxes.append((cls_id, x1, y1, width, height))
    return boxes

# Sample random images with annotations
img_paths = list((DATA_ROOT/'train'/'images').glob('*'))
samples   = random.sample(img_paths, min(12, len(img_paths)))

colors = plt.cm.tab10(np.linspace(0, 1, N_CLASSES))

fig, axes = plt.subplots(3, 4, figsize=(20, 14))
fig.suptitle('PCB Defect Dataset — Ground Truth Annotations\n'
             'Inspect carefully before training: annotation quality determines model quality',
             fontsize=13, fontweight='bold')

for ax, img_path in zip(axes.flat, samples):
    img       = cv2.imread(str(img_path))
    img_rgb   = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w      = img.shape[:2]
    label_path = DATA_ROOT/'train'/'labels'/(img_path.stem + '.txt')
    boxes      = load_yolo_boxes(label_path, w, h)
    
    ax.imshow(img_rgb)
    for cls_id, x1, y1, bw, bh in boxes:
        color = colors[cls_id % len(colors)]
        rect  = patches.Rectangle((x1,y1), bw, bh,
                                   linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1-5, CLASS_NAMES[cls_id],
                color='white', fontsize=8, fontweight='bold',
                bbox=dict(facecolor=color, alpha=0.7, pad=1))
    ax.set_title(f'{img_path.name}\n{len(boxes)} defects', fontsize=9)
    ax.axis('off')

plt.tight_layout()
plt.savefig('output/01_dataset_annotations.jpg', dpi=150, bbox_inches='tight')
plt.show()

# =============================================
# DATASET STATISTICS — class balance for detection
# =============================================

class_counts = np.zeros(N_CLASSES, dtype=int)
box_sizes    = []

for label_path in (DATA_ROOT/'train'/'labels').glob('*.txt'):
    # Get corresponding image size
    img_path = DATA_ROOT/'train'/'images'/(label_path.stem + '.jpg')
    if not img_path.exists():
        img_path = DATA_ROOT/'train'/'images'/(label_path.stem + '.png')
    if img_path.exists():
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
    else:
        w, h = 640, 640

    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_counts[int(parts[0])] += 1
                box_w = float(parts[3]) * w
                box_h = float(parts[4]) * h
                box_sizes.append(np.sqrt(box_w * box_h))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('PCB Dataset Statistics — Know Your Data Before Training',
             fontsize=12, fontweight='bold')

bars = ax1.bar(CLASS_NAMES, class_counts,
               color=[colors[i] for i in range(N_CLASSES)])
ax1.set_title('Class distribution (annotation count)')
ax1.set_ylabel('Number of annotations')
ax1.set_xlabel('Defect type')
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha='right')
for bar, count in zip(bars, class_counts):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2,
             str(count), ha='center', fontsize=10, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

ax2.hist(box_sizes, bins=40, color='#1D9E75', alpha=0.8, edgecolor='white')
ax2.axvline(np.median(box_sizes), color='red', linestyle='--',
            linewidth=2, label=f'Median: {np.median(box_sizes):.0f}px')
ax2.set_title('Defect bounding box size distribution\n(√area in pixels)')
ax2.set_xlabel('Box size (pixels)')
ax2.set_ylabel('Count')
ax2.legend(); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output/01_dataset_statistics.jpg', dpi=150)
plt.show()

# Imbalance warning
max_count, min_count = class_counts.max(), class_counts.min()
imbalance_ratio      = max_count / (min_count + 1e-8)
print(f"\nClass imbalance ratio: {imbalance_ratio:.1f}x")
if imbalance_ratio > 5:
    print("  ⚠ High imbalance — consider class weights or oversampling rare classes")
else:
    print("  ✓ Acceptable balance — standard training should work fine")
print(f"\nMedian defect size: {np.median(box_sizes):.0f}px")
print("  → If median < 32px: use smaller model stride or higher resolution input")