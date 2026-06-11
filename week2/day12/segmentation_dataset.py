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


# ============================================================
# MVTEC SEGMENTATION DATASET
# ============================================================

class MVTecSegDataset(Dataset):
    """
    Loads MVTec AD images with their binary segmentation masks.
    For training: loads good images only (mask = all zeros)
    For validation: loads defective images with ground truth masks
    """
    def __init__(self, data_root, split='train',
                 defect_types=None, transform=None, img_size=256):
        self.data_root    = Path(data_root)
        self.split        = split
        self.transform    = transform
        self.img_size     = img_size
        self.samples      = []  # list of (img_path, mask_path_or_None)

        if split == 'train':
            # Only good images — masks are all zeros
            good_path = self.data_root / 'train' / 'good'
            for img_path in sorted(good_path.glob('*.png')):
                self.samples.append((img_path, None))

        elif split == 'val':
            # Defective images with ground truth masks
            test_path = self.data_root / 'test'
            gt_path   = self.data_root / 'ground_truth'

            if defect_types is None:
                defect_types = [
                    d.name for d in test_path.iterdir()
                    if d.is_dir() and d.name != 'good'
                ]

            for dtype in defect_types:
                for img_path in sorted((test_path / dtype).glob('*.png')):
                    mask_path = gt_path / dtype / (img_path.stem + '_mask.png')
                    if mask_path.exists():
                        self.samples.append((img_path, mask_path))

        print(f"  {split}: {len(self.samples)} samples loaded")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        # Load image
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Load mask
        if mask_path is not None and Path(mask_path).exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            mask = (mask > 127).astype(np.float32)
        else:
            # Good image — mask is all zeros (no defects)
            mask = np.zeros(img.shape[:2], dtype=np.float32)

        # Apply transforms — both image and mask transformed identically
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img  = augmented['image']
            mask = augmented['mask'].unsqueeze(0)  # add channel dim → [1, H, W]
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
    train_dataset = MVTecSegDataset(
        DATA_ROOT, split='train',
        transform=get_train_transform(IMG_SIZE)
    )
    val_dataset = MVTecSegDataset(
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