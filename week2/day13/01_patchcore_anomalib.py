"""
Day 13 - Script 1: PatchCore with anomalib
Intel's production-grade anomaly detection library.
Achieves near-perfect AUROC on MVTec with zero training in the traditional sense.

INI BERMASALAH anomalib ga compatible dengan versi package yang terbaru
"""
from anomalib.data.utils import TestSplitMode, ValSplitMode
from anomalib import TaskType
from anomalib.data import MVTec
from anomalib.models import Patchcore
from anomalib.engine import Engine
from anomalib.loggers import AnomalibWandbLogger
from pathlib import Path
import os

os.environ["WANDB_MODE"] = "offline"

DATA_ROOT = Path('/Users/alvi/OpenCV/data')

print("=" * 55)
print("PatchCore Anomaly Detection on MVTec Leather")
print("=" * 55)

# ============================================================
# DATA MODULE
# anomalib handles the MVTec folder structure automatically
# ============================================================
datamodule = MVTec(
    root=str(DATA_ROOT),
    category='leather',       # the category we used in Day 12
    image_size=256,
    train_batch_size=32,
    eval_batch_size=32,
    task=TaskType.SEGMENTATION,  # outputs both score + pixel map
)
# ============================================================
# MODEL — PatchCore configuration
# ============================================================
model = Patchcore(
    backbone='wide_resnet50_2',   # stronger backbone than ResNet34
    layers=['layer2', 'layer3'],  # which ResNet layers to extract from
    # layer2: mid-level features (textures, patterns)
    # layer3: high-level features (semantic content)
    # combining both = richer patch representation

    coreset_sampling_ratio=0.1,   # keep 10% of patches (memory efficiency)
    # coreset subsampling: instead of storing ALL patch features (millions),
    # use greedy algorithm to find the most representative 10%
    # This is why PatchCore is memory-efficient

    num_neighbors=9,              # k-NN: compare to 9 nearest good patches
)

# ============================================================
# ENGINE — handles training, evaluation, logging
# ============================================================
engine = Engine(
    max_epochs=1,       # PatchCore doesn't train — it just extracts features
                        # "1 epoch" = one pass to build the memory bank
    accelerator='auto', # automatically uses MPS/GPU/CPU
    logger=False,       # set to AnomalibWandbLogger() if W&B is working
)

print("\nBuilding PatchCore memory bank...")
print("(This is feature extraction, not gradient descent — very fast)")
engine.fit(model, datamodule=datamodule)

print("\nEvaluating on test set...")
test_results = engine.test(model, datamodule=datamodule)

print("\n" + "="*55)
print("PATCHCORE RESULTS")
print("="*55)
for key, val in test_results[0].items():
    print(f"  {key}: {val:.4f}")