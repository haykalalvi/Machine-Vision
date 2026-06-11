"""
Day 12 - Script 2: Segmentation Dataset and Augmentation Pipeline
Builds a PyTorch Dataset that returns (image, binary_mask) pairs.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
import matplotlib.pyplot as plt
import random

DATA_ROOT = Path('/Users/alvi/OpenCV/data/leather')

# ============================================================
# AUGMENTATION PIPELINE — segmentation-specific
# Key difference from classification: SAME transform applied
# to BOTH image and mask — they must stay aligned
# ============================================================

def get_train_transform(img_size=256):
    return A.Compose([
        A.Resize(img_size, img_size),

        # Geometric — applied identically to image and mask
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        
        # Fixed Warning: Replaced ShiftScaleRotate with Affine
        A.Affine(
            scale=(0.9, 1.1), translate_percent=(-0.05, 0.05),
            rotate=(-15, 15), p=0.5
        ),

        # Elastic deformation — simulates surface texture variation
        A.ElasticTransform(alpha=1, sigma=50, p=0.2),

        # Photometric — applied to image ONLY (mask is binary, no color)
        A.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=0.5
        ),
        
        # Fixed Warning: Removed invalid 'var_limit' from GaussNoise
        A.GaussNoise(p=0.3),

        # Normalize with ImageNet stats (using pretrained encoder)
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def get_val_transform(img_size=256):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


"""
FIXED VERSION — MVTecSegDataset
Splits the test/ folder (which has both good + defective images with masks)
into train/val so the model actually sees defects during training.
"""


class MVTecSegDatasetFixed(Dataset):
    """
    FIXED: Uses test/ folder data (which has masks for defects)
    and splits it into train/val ourselves.

    train/good (Day 13's PatchCore data) is NOT used here —
    it has no defect examples, so it cannot teach U-Net what a defect looks like.
    """
    def __init__(self, data_root, split='train', transform=None,
                 train_ratio=0.7, seed=42):
        self.data_root = Path(data_root)
        self.transform = transform
        self.samples   = []

        test_path = self.data_root / 'test'
        gt_path   = self.data_root / 'ground_truth'

        # Collect ALL test images (good + every defect type) with their masks
        all_samples = []
        for dtype_dir in sorted(test_path.iterdir()):
            if not dtype_dir.is_dir():
                continue
            dtype = dtype_dir.name

            for img_path in sorted(dtype_dir.glob('*.png')):
                if dtype == 'good':
                    # Good images → empty mask
                    all_samples.append((img_path, None, dtype))
                else:
                    mask_path = gt_path / dtype / (img_path.stem + '_mask.png')
                    if mask_path.exists():
                        all_samples.append((img_path, mask_path, dtype))

        # Shuffle deterministically and split
        random.Random(seed).shuffle(all_samples)
        n_train = int(len(all_samples) * train_ratio)

        if split == 'train':
            self.samples = all_samples[:n_train]
        else:  # val
            self.samples = all_samples[n_train:]

        # Print class balance — important to verify
        n_good   = sum(1 for s in self.samples if s[1] is None)
        n_defect = len(self.samples) - n_good
        print(f"  {split}: {len(self.samples)} samples "
              f"({n_good} good, {n_defect} defective)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, dtype = self.samples[idx]

        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if mask_path is not None:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            mask = (mask > 127).astype(np.float32)
        else:
            mask = np.zeros(img.shape[:2], dtype=np.float32)

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img  = augmented['image']
            mask = augmented['mask'].unsqueeze(0)
        else:
            img  = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).unsqueeze(0)

        return img, mask

def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return torch.clamp(tensor * std + mean, 0, 1)


# ============================================================
# MAIN EXECUTION BLOCK (Fixes the Multiprocessing Error)
# ============================================================
if __name__ == '__main__':
    IMG_SIZE = 256

    print("Loading MVTec Leather datasets...")
    train_dataset = MVTecSegDatasetFixed(
        DATA_ROOT, split='train',
        transform=get_train_transform(IMG_SIZE)
    )
    val_dataset = MVTecSegDatasetFixed(
        DATA_ROOT, split='val',
        transform=get_val_transform(IMG_SIZE)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=8, shuffle=True,
        num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=8, shuffle=False,
        num_workers=2, pin_memory=True
    )

    print(f"\nDataset summary:")
    print(f"  Train (good only): {len(train_dataset)} images")
    print(f"  Val (defective):   {len(val_dataset)} images")

    # ============================================================
    # VERIFY: display augmented image + mask pairs
    # ============================================================

    # Get a batch from val loader (has masks)
    imgs, masks = next(iter(val_loader))
    print(f"\nBatch shapes:")
    print(f"  Images: {imgs.shape}   → [batch, channels, H, W]")
    print(f"  Masks:  {masks.shape}  → [batch, 1, H, W]")
    print(f"  Mask values: min={masks.min():.0f}, max={masks.max():.0f} (binary: 0 or 1)")

    fig, axes = plt.subplots(3, 4, figsize=(18, 13))
    fig.suptitle(
        'Segmentation Dataset Verification\n'
        'Each column: original | mask | overlay | augmented version',
        fontsize=13, fontweight='bold'
    )

    for col in range(4):
        img_disp  = denormalize(imgs[col]).permute(1, 2, 0).numpy()
        mask_disp = masks[col, 0].numpy()

        # Overlay
        overlay = img_disp.copy()
        overlay[mask_disp > 0.5] = [1.0, 0.2, 0.2]
        blended  = 0.6 * img_disp + 0.4 * overlay

        axes[0][col].imshow(img_disp)
        axes[0][col].set_title(f'Image {col+1}', fontsize=10)
        axes[0][col].axis('off')

        axes[1][col].imshow(mask_disp, cmap='gray', vmin=0, vmax=1)
        pct = mask_disp.mean() * 100
        axes[1][col].set_title(f'Mask ({pct:.1f}% defect)', fontsize=10)
        axes[1][col].axis('off')

        axes[2][col].imshow(blended)
        axes[2][col].set_title('Overlay', fontsize=10)
        axes[2][col].axis('off')

    plt.tight_layout()
    
    # Ensure the output directory exists before saving
    Path('output').mkdir(exist_ok=True)
    plt.savefig('output/02_dataset_verification.jpg', dpi=150)
    plt.show()
    print("Saved: output/02_dataset_verification.jpg")

    # Save dataset objects for use in training script
    torch.save({
        'train_dataset': train_dataset,
        'val_dataset':   val_dataset,
    }, 'output/datasets.pt')
    print("Datasets saved for reuse in training script")