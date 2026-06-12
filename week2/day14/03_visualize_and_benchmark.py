"""
Day 14 - Script 3: Visualize pipeline output and benchmark performance
"""

import sys
sys.path.append('.')
from refer_02_two_stage_pipeline import TwoStageDefectPipeline

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random
import time

YOLO_PATH = '/Users/alvi/OpenCV/week2/day14/runs/stage1_detector/weights/best.pt'
UNET_PATH = '/Users/alvi/OpenCV/output/best_unet.pt'
DATA_ROOT = Path('/Users/alvi/OpenCV/data/leather')

pipeline = TwoStageDefectPipeline(YOLO_PATH, UNET_PATH)

# ============================================================
# COLLECT TEST IMAGES — across all defect types
# ============================================================
test_path    = DATA_ROOT / 'test'
defect_types = sorted([d.name for d in test_path.iterdir() if d.is_dir() and d.name != 'good'])

samples = []
for dtype in defect_types:
    imgs = sorted((test_path / dtype).glob('*.png'))
    if imgs:
        samples.append((random.choice(imgs), dtype))

print(f"Running pipeline on {len(samples)} samples (one per defect type)...")

# ============================================================
# VISUALIZE: original | detection | full mask | overlay
# ============================================================
n = len(samples)
fig, axes = plt.subplots(n, 4, figsize=(18, 4.5*n))
fig.suptitle(
    'Two-Stage Pipeline: YOLOv8 Detection -> U-Net Segmentation\n'
    'Stage 1 finds the box (red), Stage 2 refines pixels within it (yellow overlay)',
    fontsize=13, fontweight='bold'
)

col_titles = ['Input image', 'Stage 1: detection', 'Stage 2: segmentation mask', 'Combined overlay']
for col, title in enumerate(col_titles):
    axes[0][col].set_title(title, fontsize=10, fontweight='bold')

all_timings = []

for row, (img_path, true_dtype) in enumerate(samples):
    img = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = pipeline.predict(img)
    all_timings.append(result['timing'])

    # Col 0: original
    axes[row][0].imshow(img_rgb)
    axes[row][0].set_ylabel(true_dtype, fontsize=11, fontweight='bold')
    axes[row][0].axis('off')

    # Col 1: detection boxes
    det_vis = img_rgb.copy()
    for det in result['detections']:
        x1, y1, x2, y2 = det['bbox']
        cv2.rectangle(det_vis, (x1,y1), (x2,y2), (255,0,0), 2)
        label = f"{det['class']} {det['confidence']:.0%}"
        cv2.putText(det_vis, label, (x1, max(y1-6,0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
    axes[row][1].imshow(det_vis)
    n_det = len(result['detections'])
    axes[row][1].set_title(f'{n_det} detection(s)', fontsize=9) if row==0 else None
    axes[row][1].axis('off')

    # Col 2: full segmentation mask
    axes[row][2].imshow(result['full_mask'], cmap='gray')
    axes[row][2].axis('off')

    # Col 3: overlay
    overlay = img_rgb.copy()
    mask_bool = result['full_mask'] > 0
    overlay[mask_bool] = [255, 220, 0]  # yellow for defect pixels
    blended = cv2.addWeighted(img_rgb, 0.6, overlay, 0.4, 0)
    for det in result['detections']:
        x1, y1, x2, y2 = det['bbox']
        cv2.rectangle(blended, (x1,y1), (x2,y2), (255,0,0), 2)
    axes[row][3].imshow(blended)
    if result['detections']:
        area_pct = result['detections'][0]['defect_area_pct']
        axes[row][3].set_title(f'Defect area: {area_pct:.1f}% of crop',
                               fontsize=9) if row==0 else None
    axes[row][3].axis('off')

plt.tight_layout()
plt.savefig('output/03_pipeline_visualization.jpg', dpi=120, bbox_inches='tight')
plt.show()
print("Saved: output/03_pipeline_visualization.jpg")

# ============================================================
# FULL BENCHMARK — average over more images
# ============================================================
print("\n" + "="*55)
print("FULL PIPELINE BENCHMARK")
print("="*55)

bench_imgs = []
for dtype in defect_types + ['good']:
    bench_imgs.extend(sorted((test_path / dtype).glob('*.png'))[:10])

print(f"Benchmarking on {len(bench_imgs)} images...")

timings = {'detect_ms': [], 'segment_ms': [], 'total_ms': []}
detection_counts = []

for img_path in bench_imgs:
    img = cv2.imread(str(img_path))
    result = pipeline.predict(img)
    for k in timings:
        timings[k].append(result['timing'][k])
    detection_counts.append(len(result['detections']))

print(f"\nAverage timing over {len(bench_imgs)} images:")
print(f"  Stage 1 (detect):  {np.mean(timings['detect_ms']):.2f} ms")
print(f"  Stage 2 (segment): {np.mean(timings['segment_ms']):.2f} ms  "
      f"(only when detections found)")
print(f"  Total:             {np.mean(timings['total_ms']):.2f} ms")
print(f"  Estimated FPS:     {1000/np.mean(timings['total_ms']):.1f}")
print(f"\nAverage detections per image: {np.mean(detection_counts):.2f}")

# Histogram of timings
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Two-Stage Pipeline Performance', fontsize=13, fontweight='bold')

axes[0].hist(timings['total_ms'], bins=20, color='#1D9E75', alpha=0.8, edgecolor='white')
axes[0].axvline(np.mean(timings['total_ms']), color='red', linestyle='--',
                label=f"Mean: {np.mean(timings['total_ms']):.1f}ms")
axes[0].set_title('Total latency distribution')
axes[0].set_xlabel('Time (ms)'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

stages  = ['Stage 1\n(YOLOv8)', 'Stage 2\n(U-Net)']
means   = [np.mean(timings['detect_ms']), np.mean(timings['segment_ms'])]
axes[1].bar(stages, means, color=['#378ADD', '#7F77DD'])
axes[1].set_title('Time breakdown by stage')
axes[1].set_ylabel('Time (ms)')
for i, m in enumerate(means):
    axes[1].text(i, m+0.5, f'{m:.1f}ms', ha='center', fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('output/03_benchmark.jpg', dpi=150)
plt.show()
print("\nSaved: output/03_benchmark.jpg")

required_fps = 10
achieved_fps = 1000/np.mean(timings['total_ms'])
print(f"\n{'PASS' if achieved_fps > required_fps else 'NEEDS OPTIMIZATION'}: "
      f"{achieved_fps:.1f} FPS vs {required_fps} FPS target")