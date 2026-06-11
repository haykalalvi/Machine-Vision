from ultralytics import YOLO
import wandb
from pathlib import Path
import yaml

# =============================================
# YOLO MODEL SIZE GUIDE — choose based on your hardware
# =============================================
# YOLOv8n (nano)   → fastest, smallest, lowest accuracy → use if CPU only
# YOLOv8s (small)  → good balance → recommended starting point
# YOLOv8m (medium) → higher accuracy → use with Colab GPU
# YOLOv8l (large)  → high accuracy, slow → production with dedicated GPU
# YOLOv8x (xlarge) → highest accuracy, slowest → competition/benchmarking

DATA_YAML = '/Users/alvi/OpenCV/data/PCB Defect/data.yaml'
MODEL_SIZE = 'yolov8s'   # start with small — upgrade after first results

# =============================================
# EXPERIMENT 1: BASELINE — default hyperparameters
# Always establish a baseline before tuning
# =============================================

print("="*55)
print("EXPERIMENT 1: Baseline YOLOv8s")
print("="*55)

wandb.init(project="day11-pcb-yolov8",
           name=f"{MODEL_SIZE}-baseline",
           config={'model': MODEL_SIZE, 'epochs': 50, 'strategy': 'baseline'})

model_baseline = YOLO(f'{MODEL_SIZE}.pt')  # loads pretrained COCO weights

results_baseline = model_baseline.train(
    data=DATA_YAML,
    epochs=50,
    imgsz=640,          # standard input resolution
    batch=16,           # reduce to 8 if out of memory
    project='runs/pcb',
    name='baseline',
    
    # Optimization
    optimizer='AdamW',
    lr0=1e-3,           # initial learning rate
    lrf=0.01,           # final LR = lr0 * lrf
    warmup_epochs=3,    # gradual LR warmup — important for fine-tuning
    
    # Regularization
    weight_decay=5e-4,
    dropout=0.0,        # YOLOv8 supports dropout in classifier
    
    # Augmentation — built into YOLO
    mosaic=1.0,         # mosaic augmentation: combines 4 images into 1
    mixup=0.1,          # mixup: blends 2 images — helps generalization
    copy_paste=0.1,     # copies objects between images — great for rare defects
    
    # Hardware
    device=0 if __import__('torch').cuda.is_available() else ('mps' if __import__('torch').backends.mps.is_available() else 'cpu'),
    workers=4,
    
    # Logging
    save=True,
    plots=True,
    verbose=True,
)

wandb.finish()
print(f"\nBaseline mAP50: {results_baseline.results_dict.get('metrics/mAP50(B)', 'N/A')}")
print(f"Baseline mAP50-95: {results_baseline.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")