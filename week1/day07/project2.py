import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import seaborn as sns

DATASET_PATH = Path('/Users/alvi/OpenCV/week1/day07/data/hazelnut')  # adjust to your path

# =============================================
# STEP 1: BUILD A REFERENCE FROM GOOD IMAGES
# =============================================

def build_reference(good_images_path):
    """Compute mean and std of all good training images"""
    images = []
    for img_path in sorted(good_images_path.glob('*.png')):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (256, 256))
        images.append(img.astype(np.float32))
    
    stack = np.stack(images, axis=0)
    mean_img = stack.mean(axis=0)
    std_img  = stack.std(axis=0)
    
    print(f"Reference built from {len(images)} good training images")
    return mean_img, std_img

good_train_path = DATASET_PATH / 'train' / 'good'
ref_mean, ref_std = build_reference(good_train_path)

# Visualize the reference
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.imshow(ref_mean, cmap='gray')
ax1.set_title('Mean reference image\n(average of all good samples)')
ax1.axis('off')
ax2.imshow(ref_std, cmap='hot')
ax2.set_title('Std deviation map\n(bright = high natural variation)')
ax2.axis('off')
plt.tight_layout()
plt.savefig('output/reference_images.jpg', dpi=150)
# plt.show()

# =============================================
# STEP 2: COMPUTE ANOMALY SCORE PER IMAGE
# =============================================

def compute_anomaly_score(img_bgr, ref_mean, ref_std, threshold=0.5):
    """
    Compare test image against reference.
    Returns: anomaly_score, anomaly_map, binary_mask
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = cv2.resize(gray, (256, 256))
    
    # Normalize deviation by expected std (z-score per pixel)
    epsilon = 1e-8  # avoid division by zero
    z_map = np.abs(gray - ref_mean) / (ref_std + epsilon)
    
    # Smooth to remove noise
    z_map_smooth = cv2.GaussianBlur(z_map, (15, 15), 0)
    
    # Global anomaly score = 95th percentile of the deviation map
    # (more robust than max, less affected by single noisy pixels)
    anomaly_score = float(np.percentile(z_map_smooth, 95))
    
    # Binary mask: pixels that deviate more than threshold
    binary_mask = (z_map_smooth > threshold).astype(np.uint8) * 255
    
    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    
    return anomaly_score, z_map_smooth, binary_mask

# =============================================
# STEP 3: RUN ON ALL TEST IMAGES & EVALUATE
# =============================================

all_scores, all_labels = [], []
results_sample = []

test_path = DATASET_PATH / 'test'
SCORE_THRESHOLD = 1.5  # tune this — if too many false positives, increase

for defect_dir in sorted(test_path.iterdir()):
    if not defect_dir.is_dir():
        continue
    
    is_defect = defect_dir.name != 'good'
    label = 1 if is_defect else 0
    
    for img_path in sorted(defect_dir.glob('*.png')):
        img = cv2.imread(str(img_path))
        score, z_map, mask = compute_anomaly_score(img, ref_mean, ref_std)
        
        all_scores.append(score)
        all_labels.append(label)
        
        # Save a few samples for visualization
        if len(results_sample) < 12:
            results_sample.append({
                'img': img, 'score': score, 'z_map': z_map,
                'mask': mask, 'label': label,
                'type': defect_dir.name,
                'pred': 1 if score > SCORE_THRESHOLD else 0
            })

# =============================================
# STEP 4: METRICS
# =============================================

all_preds = [1 if s > SCORE_THRESHOLD else 0 for s in all_scores]

auroc = roc_auc_score(all_labels, all_scores)
print("\n=== DETECTION RESULTS ===")
print(f"AUROC Score:       {auroc:.4f}  (1.0 = perfect, 0.5 = random)")
print(f"Score Threshold:   {SCORE_THRESHOLD}")
print(f"\n{classification_report(all_labels, all_preds, target_names=['Good', 'Defect'])}")

cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Pred: Good', 'Pred: Defect'],
            yticklabels=['True: Good', 'True: Defect'])
ax.set_title(f'Confusion Matrix\nAUROC = {auroc:.4f}')
plt.tight_layout()
plt.savefig('output/confusion_matrix.jpg', dpi=150)
# plt.show()

# =============================================
# STEP 5: VISUALIZE SAMPLE PREDICTIONS
# =============================================

fig, axes = plt.subplots(4, 9, figsize=(22, 10))
fig.suptitle(f'Classical Defect Detector Results | AUROC={auroc:.3f}', fontsize=13)

for i, r in enumerate(results_sample[:12]):
    row = i // 3
    col = (i % 3) * 3
    
    # Original image
    axes[row][col].imshow(cv2.cvtColor(r['img'], cv2.COLOR_BGR2RGB))
    status = "✓ PASS" if r['pred'] == 0 else "✗ FAIL"
    correct = r['pred'] == r['label']
    color = 'green' if correct else 'red'
    axes[row][col].set_title(f"{r['type']}\n{status} (score:{r['score']:.2f})", 
                              color=color, fontsize=8)
    axes[row][col].axis('off')
    
    # Z-score deviation map
    axes[row][col+1].imshow(r['z_map'], cmap='hot')
    axes[row][col+1].set_title('Deviation map', fontsize=8)
    axes[row][col+1].axis('off')
    
    # Binary mask overlay
    # overlay = r['img'].copy()
    overlay = cv2.resize(r['img'], (256, 256))
    overlay[r['mask'] > 0] = [0, 0, 255]
    axes[row][col+2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[row][col+2].set_title('Defect mask', fontsize=8)
    axes[row][col+2].axis('off')

plt.tight_layout()
plt.savefig('output/detection_results.jpg', dpi=150, bbox_inches='tight')
# plt.show()