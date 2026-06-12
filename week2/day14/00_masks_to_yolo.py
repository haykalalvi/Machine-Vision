"""
Day 14 - Script 0: Convert MVTec ground truth masks to YOLO bounding boxes
This lets us train YOLOv8 on the SAME dataset/defect-types as our Day 12 U-Net,
so Stage 1 (detect) and Stage 2 (segment) operate on consistent defect classes.
"""

import cv2
import numpy as np
from pathlib import Path
import yaml
import shutil

DATA_ROOT = Path('/Users/alvi/OpenCV/data/leather')
OUTPUT    = Path('/Users/alvi/OpenCV/week2/day14/data/leather_yolo')

# Defect types = our YOLO classes
defect_types = sorted([
    d.name for d in (DATA_ROOT / 'ground_truth').iterdir() if d.is_dir()
])
CLASS_TO_ID = {name: i for i, name in enumerate(defect_types)}
print(f"Classes: {CLASS_TO_ID}")

def mask_to_boxes(mask, min_area=20):
    """
    Find connected components in a binary mask and return bounding boxes.
    A single mask can contain multiple disconnected defect regions.
    """
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue  # skip tiny noise regions
        x, y, w, h = cv2.boundingRect(c)
        boxes.append((x, y, w, h))
    return boxes

# ============================================================
# BUILD YOLO DATASET STRUCTURE
# ============================================================
for split in ['train', 'val']:
    (OUTPUT / split / 'images').mkdir(parents=True, exist_ok=True)
    (OUTPUT / split / 'labels').mkdir(parents=True, exist_ok=True)

# Use the SAME train/val split logic as Day 12's fix:
# combine all test/ images (good + defective) and split 70/30
import random
all_samples = []
test_path = DATA_ROOT / 'test'
gt_path   = DATA_ROOT / 'ground_truth'

for dtype_dir in sorted(test_path.iterdir()):
    if not dtype_dir.is_dir():
        continue
    dtype = dtype_dir.name
    for img_path in sorted(dtype_dir.glob('*.png')):
        if dtype == 'good':
            all_samples.append((img_path, None, dtype))
        else:
            mask_path = gt_path / dtype / (img_path.stem + '_mask.png')
            if mask_path.exists():
                all_samples.append((img_path, mask_path, dtype))

random.Random(42).shuffle(all_samples)
n_train = int(len(all_samples) * 0.7)
splits  = {'train': all_samples[:n_train], 'val': all_samples[n_train:]}

# ============================================================
# CONVERT EACH IMAGE
# ============================================================
for split_name, samples in splits.items():
    n_with_boxes = 0
    for img_path, mask_path, dtype in samples:
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]

        # Copy image
        out_img_name = f"{dtype}_{img_path.stem}.png"
        cv2.imwrite(str(OUTPUT / split_name / 'images' / out_img_name), img)

        # Build label file (empty for 'good' images)
        label_lines = []
        if mask_path is not None:
            mask  = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            mask  = (mask > 127).astype(np.uint8) * 255
            boxes = mask_to_boxes(mask)
            class_id = CLASS_TO_ID[dtype]

            for (x, y, bw, bh) in boxes:
                # Convert to YOLO normalized format: cx, cy, w, h
                cx = (x + bw/2) / w
                cy = (y + bh/2) / h
                nw = bw / w
                nh = bh / h
                label_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            if boxes:
                n_with_boxes += 1

        label_path = OUTPUT / split_name / 'labels' / f"{out_img_name[:-4]}.txt"
        with open(label_path, 'w') as f:
            f.write('\n'.join(label_lines))

    print(f"{split_name}: {len(samples)} images, {n_with_boxes} with defect boxes")

# ============================================================
# WRITE data.yaml
# ============================================================
yaml_content = {
    'path':  str(OUTPUT.resolve()),
    'train': 'train/images',
    'val':   'val/images',
    'nc':    len(defect_types),
    'names': {i: name for i, name in enumerate(defect_types)},
}
with open(OUTPUT / 'data.yaml', 'w') as f:
    yaml.dump(yaml_content, f, default_flow_style=False)

print(f"\ndata.yaml written to {OUTPUT/'data.yaml'}")
print(f"Classes: {defect_types}")
print("\n✓ Ready for Stage 1 YOLOv8 training")