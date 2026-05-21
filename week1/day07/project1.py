import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATASET_PATH = Path('/Users/alvi/OpenCV/week1/day07/data/hazelnut')  # adjust to your path

def load_mvtec_split(category_path, split='test'):
    """Load images and labels from MVTec AD dataset structure"""
    images, labels, defect_types = [], [], []
    split_path = category_path / split
    
    for defect_dir in sorted(split_path.iterdir()):
        if not defect_dir.is_dir():
            continue
        defect_name = defect_dir.name
        label = 0 if defect_name == 'good' else 1
        
        for img_path in sorted(defect_dir.glob('*.png')):
            img = cv2.imread(str(img_path))
            img = cv2.resize(img, (256, 256))
            images.append(img)
            labels.append(label)
            defect_types.append(defect_name)
    
    return images, labels, defect_types

# Load test set
test_imgs, test_labels, test_types = load_mvtec_split(DATASET_PATH, 'test')

print("=== Dataset Statistics ===")
print(f"Total test images: {len(test_imgs)}")
print(f"Good: {test_labels.count(0)}")
print(f"Defective: {test_labels.count(1)}")
print(f"Defect types: {set(t for t in test_types if t != 'good')}")

# Display sample grid
fig, axes = plt.subplots(3, 6, figsize=(18, 9))
fig.suptitle('MVTec AD Dataset Samples', fontsize=14, fontweight='bold')

shown = {}
idx = 0
for i, (img, label, dtype) in enumerate(zip(test_imgs, test_labels, test_types)):
    if dtype not in shown and idx < 18:
        ax = axes.flat[idx]
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        color = 'red' if label == 1 else 'green'
        ax.set_title(dtype, color=color, fontsize=9)
        ax.axis('off')
        shown[dtype] = True
        idx += 1

plt.tight_layout()
plt.savefig('output/dataset_overview.jpg', dpi=150)
plt.show()