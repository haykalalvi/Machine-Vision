"""
Day 14 - Script 1: Train Stage 1 YOLOv8 on MVTec leather defect boxes
Quick training — 30 epochs, nano model. The goal is "good enough to crop",
not state-of-the-art detection. Stage 2 (U-Net) handles precision.
"""

from ultralytics import YOLO
import torch

DATA_YAML = '/Users/alvi/OpenCV/week2/day14/data/leather_yolo/data.yaml' # the data that specially created for only day 14
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'

print(f"Device: {DEVICE}")
print("Training Stage 1 detector (YOLOv8n, 30 epochs)...")
print("Goal: locate defect regions for cropping, not final precision\n")

model = YOLO('yolov8n.pt')

results = model.train(
    data=DATA_YAML,
    epochs=30,
    imgsz=512,
    batch=8,
    project='/Users/alvi/OpenCV/week2/day14/runs',
    name='stage1_detector',
    device=DEVICE,
    optimizer='AdamW',
    lr0=1e-3,
    warmup_epochs=3,
    mosaic=1.0,
    plots=True,
    verbose=True,
)

# Quick eval
metrics = model.val()
print(f"\n=== Stage 1 Results ===")
print(f"mAP50:    {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
print(f"Recall:   {metrics.box.mr:.4f}")
print("\nFor a cropping pre-stage, prioritize RECALL —")
print("a missed crop means U-Net never sees that defect at all.")
print(f"\nBest weights: runs/stage1_detector/weights/best.pt")