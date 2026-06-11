"""
Day 13 - Script 2: PatchCore Implementation From Scratch
No anomalib — we build every step manually so you truly understand it.

Pipeline:
1. Extract patch features from good training images using pretrained ResNet
2. Build memory bank with coreset subsampling
3. For each test image: compute anomaly score + pixel-level anomaly map
4. Evaluate with AUROC
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import time
from sklearn.random_projection import SparseRandomProjection


DATA_ROOT = Path('/Users/alvi/OpenCV/data/leather')
DEVICE    = torch.device(
    'mps'  if torch.backends.mps.is_available() else
    'cuda' if torch.cuda.is_available() else
    'cpu'
)
IMG_SIZE  = 256
print(f"Device: {DEVICE}")

# ============================================================
# STEP 1: FEATURE EXTRACTOR
# Use pretrained ResNet50 — extract intermediate layer features
# NOT the final classification layer — we want spatial features
# ============================================================

class PatchFeatureExtractor(nn.Module):
    """
    Extracts patch-level features from two intermediate ResNet layers.
    These features represent local image patches — not global semantics.

    Why intermediate layers?
    - Early layers: too low-level (edges only)
    - Final layer: too high-level (lost spatial information)
    - Middle layers: texture + structure = ideal for anomaly detection
    """
    def __init__(self):
        super().__init__()
        resnet = models.wide_resnet50_2(weights='IMAGENET1K_V1')

        # Extract feature maps before pooling
        self.layer1 = nn.Sequential(*list(resnet.children())[:5])  # 64×64
        self.layer2 = nn.Sequential(*list(resnet.children())[:6])  # 32×32
        self.layer3 = nn.Sequential(*list(resnet.children())[:7])  # 16×16

        # Freeze all parameters — we're not training, only extracting
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        _      = self.layer1(x)
        feat2  = self.layer2(x)  # [B, 512, 32, 32]
        feat3  = self.layer3(x)  # [B, 1024, 16, 16]

        # Upsample layer3 to match layer2 spatial size
        feat3_up = nn.functional.interpolate(
            feat3, size=feat2.shape[-2:], mode='bilinear', align_corners=False
        )

        # Concatenate along channel dimension
        # Combined: [B, 1536, 32, 32]
        combined = torch.cat([feat2, feat3_up], dim=1)
        return combined


extractor = PatchFeatureExtractor().to(DEVICE)
extractor.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


def load_images_from_folder(folder_path, max_imgs=None):
    """Load all PNG images from a folder as numpy arrays"""
    imgs = []
    paths = sorted(Path(folder_path).glob('*.png'))
    if max_imgs:
        paths = paths[:max_imgs]
    for p in paths:
        img = cv2.imread(str(p))
        if img is not None:
            imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return imgs


def extract_patch_features(images, extractor, transform, device, batch_size=8):
    """
    Extract patch features from a list of images.
    Returns: numpy array of shape [N_patches_total, feature_dim]
    """
    all_features = []

    for i in tqdm(range(0, len(images), batch_size), desc="Extracting features"):
        batch_imgs = images[i:i+batch_size]
        tensors    = torch.stack([transform(img) for img in batch_imgs]).to(device)

        with torch.no_grad():
            features = extractor(tensors)  # [B, 1536, 32, 32]

        # Reshape: [B, C, H, W] → [B*H*W, C]
        B, C, H, W = features.shape
        features   = features.permute(0, 2, 3, 1)  # [B, H, W, C]
        features   = features.reshape(-1, C)        # [B*H*W, C]

        all_features.append(features.cpu().numpy())

    return np.concatenate(all_features, axis=0)


# ============================================================
# STEP 2: BUILD MEMORY BANK FROM GOOD TRAINING IMAGES
# ============================================================

print("\n" + "="*55)
print("STEP 2: Building memory bank from good training images")
print("="*55)

train_images = load_images_from_folder(DATA_ROOT / 'train' / 'good')
print(f"Good training images: {len(train_images)}")

t0 = time.time()
train_features = extract_patch_features(train_images, extractor, transform, DEVICE)
print(f"Memory bank shape: {train_features.shape}")
print(f"  = {len(train_images)} images × {32*32} patches × {train_features.shape[1]} features")
print(f"Feature extraction time: {time.time()-t0:.1f}s")

# ============================================================
# STEP 3: CORESET SUBSAMPLING
# Storing ALL patches is memory-expensive and slow at inference.
# Coreset: find the most representative subset using greedy algorithm.
# Keep only 10% of patches while preserving coverage of the feature space.
# The Greedy Algorithm: It picks a random starting patch. 
# Then, it searches the remaining patches and picks the one that is mathematically furthest away from the first. 
# It repeats this process until it has kept exactly 10% of the patches.
# This guarantees that the 10% it keeps represents the widest possible variety of normal leather patterns, efficiently summarizing the feature space without redundancy.
# ============================================================

print("\n" + "="*55)
print("STEP 3: Coreset subsampling (10% of patches)")
print("="*55)

# from sklearn.random_projection import SparseRandomProjection

def greedy_coreset_sampling_fast(features, ratio=0.01, seed=42, projection_dim=128):
    """
    Fast greedy coreset subsampling using:
    1. Random projection to reduce dimensionality (1536 → 128)
    2. Vectorized distance updates (no recomputation from scratch)

    This matches the actual PatchCore paper's approach.
    """
    np.random.seed(seed)
    N = len(features)
    n_keep = max(1, int(N * ratio))

    print(f"  Original dim: {features.shape[1]} → projecting to {projection_dim}")

    # Step 1: Random projection — preserves distances approximately
    # but in MUCH lower dimensionality (1536 → 128 = 12x speedup per distance calc)
    projector = SparseRandomProjection(
        n_components=projection_dim, random_state=seed
    )
    features_proj = projector.fit_transform(features).astype(np.float32)

    print(f"  Selecting {n_keep:,} / {N:,} patches ({ratio:.1%})")

    # Step 2: Greedy farthest-point sampling on PROJECTED features
    selected = [np.random.randint(0, N)]
    min_dists = np.linalg.norm(
        features_proj - features_proj[selected[0]], axis=1
    )

    for _ in tqdm(range(n_keep - 1), desc="Coreset sampling"):
        next_idx = np.argmax(min_dists)
        selected.append(next_idx)

        # Only compute distance to the NEW point, then take min
        # This is O(N) per step instead of O(N) recompute — same complexity
        # but the projected dims make each step ~12x faster
        new_dists = np.linalg.norm(
            features_proj - features_proj[next_idx], axis=1
        )
        min_dists = np.minimum(min_dists, new_dists)

    return np.array(selected)

t0 = time.time()
# coreset_idx  = greedy_coreset_sampling(train_features, ratio=0.1)
# NEW
coreset_idx = greedy_coreset_sampling_fast(
    train_features, ratio=0.01, projection_dim=128
)
memory_bank = train_features[coreset_idx]
# memory_bank  = train_features[coreset_idx]
print(f"Memory bank: {len(train_features):,} → {len(memory_bank):,} patches "
      f"({len(memory_bank)/len(train_features):.0%} kept)")
print(f"Coreset sampling time: {time.time()-t0:.1f}s")

# ============================================================
# STEP 4: NEAREST NEIGHBOR INDEX
# For each test patch: find its distance to nearest good patch
# ============================================================

print("\n" + "="*55)
print("STEP 4: Building nearest neighbor index")
print("="*55)

nn_index = NearestNeighbors(n_neighbors=9, metric='euclidean', algorithm='ball_tree')
nn_index.fit(memory_bank)
print(f"NN index built on {len(memory_bank):,} memory patches")

# ============================================================
# STEP 5: INFERENCE — compute anomaly scores on test images
# ============================================================

print("\n" + "="*55)
print("STEP 5: Computing anomaly scores on test set")
print("="*55)

test_path = DATA_ROOT / 'test'
gt_path   = DATA_ROOT / 'ground_truth'

all_scores      = []  # image-level anomaly scores
all_labels      = []  # 0=good, 1=defective
all_anomaly_maps = [] # pixel-level maps
all_gt_masks    = []  # ground truth masks
all_test_imgs   = []  # for visualization

defect_types = sorted([d.name for d in test_path.iterdir() if d.is_dir()])

for dtype in defect_types:
    is_defect  = (dtype != 'good')
    test_imgs  = load_images_from_folder(test_path / dtype, max_imgs=15)

    for idx, img in enumerate(tqdm(test_imgs, desc=f"  {dtype}")):
        tensor   = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            features = extractor(tensor)    # [1, 1536, 32, 32]

        B, C, H, W = features.shape
        patches    = features.permute(0,2,3,1).reshape(-1, C).cpu().numpy()

        # Find distance to nearest neighbor in memory bank
        dists, _ = nn_index.kneighbors(patches)
        # Anomaly score per patch = distance to nearest good patch
        patch_scores = dists[:, 0]         # [H*W]

        # Image-level score = max patch score
        # (one highly anomalous patch = anomalous image)
        image_score = patch_scores.max()

        # Reshape to spatial anomaly map
        anomaly_map = patch_scores.reshape(H, W)
        # Upsample to original image size
        anomaly_map = cv2.resize(anomaly_map, (IMG_SIZE, IMG_SIZE))

        # Load ground truth mask if available
        if is_defect:
            img_paths = sorted((test_path / dtype).glob('*.png'))
            if idx < len(img_paths):
                mask_name = img_paths[idx].stem + '_mask.png'
                mask_path = gt_path / dtype / mask_name
                if mask_path.exists():
                    gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    gt_mask = cv2.resize(gt_mask, (IMG_SIZE, IMG_SIZE))
                    gt_mask = (gt_mask > 127).astype(np.uint8)
                else:
                    gt_mask = np.ones((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
            else:
                gt_mask = np.ones((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        else:
            gt_mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

        all_scores.append(image_score)
        all_labels.append(1 if is_defect else 0)
        all_anomaly_maps.append(anomaly_map)
        all_gt_masks.append(gt_mask)
        all_test_imgs.append(img)

# ============================================================
# STEP 6: NORMALIZE SCORES AND COMPUTE AUROC
# ============================================================

all_scores = np.array(all_scores)
all_labels = np.array(all_labels)

# Normalize scores to [0, 1]
score_min, score_max = all_scores.min(), all_scores.max()
norm_scores = (all_scores - score_min) / (score_max - score_min + 1e-8)

# Normalize anomaly maps
norm_maps = []
for m in all_anomaly_maps:
    m_norm = (m - score_min) / (score_max - score_min + 1e-8)
    norm_maps.append(np.clip(m_norm, 0, 1))

# Compute AUROC
auroc_image = roc_auc_score(all_labels, norm_scores)

# Pixel-level AUROC (only on defective images)
defect_mask_flat  = np.concatenate([
    m.flatten() for m, l in zip(norm_maps, all_labels) if l == 1
])
defect_score_flat = np.concatenate([
    gt.flatten() for gt, l in zip(all_gt_masks, all_labels) if l == 1
])
auroc_pixel = roc_auc_score(defect_score_flat, defect_mask_flat)

print("\n" + "="*55)
print("PATCHCORE FROM SCRATCH — RESULTS")
print("="*55)
print(f"Image-level AUROC: {auroc_image:.4f}")
print(f"Pixel-level AUROC: {auroc_pixel:.4f}")
print(f"\nInterpretation:")
print(f"  {'Excellent' if auroc_image > 0.90 else 'Good' if auroc_image > 0.80 else 'Decent'} "
      f"image-level detection")
print(f"  {'Excellent' if auroc_pixel > 0.90 else 'Good' if auroc_pixel > 0.80 else 'Decent'} "
      f"pixel-level localization")

# ============================================================
# STEP 7: VISUALIZATION
# ============================================================

# ROC Curve
fpr, tpr, thresholds = roc_curve(all_labels, norm_scores)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(f'PatchCore From Scratch — MVTec Leather\n'
             f'Image AUROC: {auroc_image:.4f} | Pixel AUROC: {auroc_pixel:.4f}',
             fontsize=13, fontweight='bold')

axes[0].plot(fpr, tpr, color='#1D9E75', linewidth=2.5, label=f'ROC (AUC={auroc_image:.3f})')
axes[0].plot([0,1],[0,1],'k--', linewidth=1.5, label='Random (AUC=0.5)')
axes[0].fill_between(fpr, tpr, alpha=0.1, color='#1D9E75')
axes[0].set_xlabel('False Positive Rate'); axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curve (Image-level)'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

# Score distribution
good_scores   = norm_scores[all_labels == 0]
defect_scores = norm_scores[all_labels == 1]
axes[1].hist(good_scores,   bins=25, alpha=0.7, color='#1D9E75', label=f'Good ({len(good_scores)})',    density=True)
axes[1].hist(defect_scores, bins=25, alpha=0.7, color='#D85A30', label=f'Defect ({len(defect_scores)})', density=True)
axes[1].set_xlabel('Normalized anomaly score'); axes[1].set_ylabel('Density')
axes[1].set_title('Score Distribution\nLess overlap = better separation')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output/02_roc_and_scores.jpg', dpi=150)
plt.show()

# Anomaly map visualization
defect_indices = [i for i, l in enumerate(all_labels) if l == 1][:8]
fig, axes = plt.subplots(len(defect_indices), 4, figsize=(18, 5*len(defect_indices)))
fig.suptitle('PatchCore Anomaly Maps — From Scratch Implementation\n'
             'No training, no labels — only good image memory',
             fontsize=13, fontweight='bold')

col_titles = ['Input image', 'Ground truth mask', 'PatchCore anomaly map', 'Threshold overlay (>0.5)']
for col, title in enumerate(col_titles):
    axes[0][col].set_title(title, fontsize=10, fontweight='bold')

for row, idx in enumerate(defect_indices):
    img      = all_test_imgs[idx]
    gt_mask  = all_gt_masks[idx]
    amap     = norm_maps[idx]
    score    = norm_scores[idx]
    amap_bin = (amap > 0.5).astype(np.uint8)

    # Colored heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * amap), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Overlay
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    overlay     = (0.5 * img_resized / 255. + 0.5 * heatmap / 255.)
    overlay     = np.clip(overlay, 0, 1)

    axes[row][0].imshow(cv2.resize(img, (IMG_SIZE, IMG_SIZE)))
    axes[row][0].set_ylabel(f'Score: {score:.3f}', fontsize=10, fontweight='bold')
    axes[row][0].axis('off')

    axes[row][1].imshow(gt_mask, cmap='gray'); axes[row][1].axis('off')
    axes[row][2].imshow(overlay); axes[row][2].axis('off')
    axes[row][3].imshow(amap_bin, cmap='hot'); axes[row][3].axis('off')

plt.tight_layout()
plt.savefig('output/02_anomaly_maps.jpg', dpi=120, bbox_inches='tight')
plt.show()
print("Saved: output/02_anomaly_maps.jpg")

# Save results for Script 3
np.save('output/patchcore_scores.npy', norm_scores)
np.save('output/patchcore_labels.npy', all_labels)
np.save('output/patchcore_maps.npy',   np.array(norm_maps))
print("\nResults saved for comparison script")
print("\n✓ PatchCore from scratch complete!")
print(f"  Image AUROC: {auroc_image:.4f}")
print(f"  Pixel AUROC: {auroc_pixel:.4f}")