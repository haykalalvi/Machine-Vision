"""
Day 15 - Diagnostic: Why does Static INT8 lose 32/48 detections?
Compares raw model OUTPUTS (before postprocessing) between FP32
and Static INT8 on the SAME preprocessed input.
"""

import onnxruntime as ort
import numpy as np
import cv2
from pathlib import Path

DATA_ROOT = Path('/Users/alvi/OpenCV/data/leather')
IMG_SIZE = 512

# Use one image that LOST detections in static quant
img_path = sorted((DATA_ROOT / 'test' / 'cut').glob('*.png'))[0]
img = cv2.imread(str(img_path))

# Preprocess EXACTLY as Script 4 does
img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
img_norm = img_rgb.astype(np.float32) / 255.0
img_chw = np.transpose(img_norm, (2, 0, 1))
input_tensor = np.expand_dims(img_chw, axis=0)

print(f"Input tensor shape: {input_tensor.shape}")
print(f"Input tensor dtype: {input_tensor.dtype}")
print(f"Input value range:  [{input_tensor.min():.4f}, {input_tensor.max():.4f}]")

# Load both models
fp32_sess = ort.InferenceSession(
    '/Users/alvi/OpenCV/week2/day14/runs/stage1_detector/weights/best.onnx',
    providers=['CPUExecutionProvider']
)
static_sess = ort.InferenceSession(
    'output/stage1_static_int8.onnx',
    providers=['CPUExecutionProvider']
)

# Check input names and shapes match
print(f"\nFP32 input:   {fp32_sess.get_inputs()[0].name}, "
      f"shape={fp32_sess.get_inputs()[0].shape}, "
      f"type={fp32_sess.get_inputs()[0].type}")
print(f"Static input: {static_sess.get_inputs()[0].name}, "
      f"shape={static_sess.get_inputs()[0].shape}, "
      f"type={static_sess.get_inputs()[0].type}")

# Run both
fp32_out = fp32_sess.run(None, {'images': input_tensor})[0]
static_out = static_sess.run(None, {'images': input_tensor})[0]

print(f"\nFP32 output shape:   {fp32_out.shape}")
print(f"Static output shape: {static_out.shape}")

print(f"\nFP32 output range:   [{fp32_out.min():.4f}, {fp32_out.max():.4f}]")
print(f"Static output range: [{static_out.min():.4f}, {static_out.max():.4f}]")

# Check the confidence/class scores specifically (rows 4 onwards)
fp32_scores = fp32_out[0, 4:, :]
static_scores = static_out[0, 4:, :]

print(f"\nFP32 max score across all anchors:   {fp32_scores.max():.4f}")
print(f"Static max score across all anchors: {static_scores.max():.4f}")

# How many anchors exceed conf_thresh=0.25 in each?
fp32_above = (fp32_scores.max(axis=0) > 0.25).sum()
static_above = (static_scores.max(axis=0) > 0.25).sum()
print(f"\nFP32 anchors above 0.25:   {fp32_above}")
print(f"Static anchors above 0.25: {static_above}")

# Check raw output magnitude difference
diff = np.abs(fp32_out - static_out)
print(f"\nMax absolute difference:  {diff.max():.4f}")
print(f"Mean absolute difference: {diff.mean():.4f}")

# Correlation -- are they even measuring the "same thing" at different scale?
fp32_flat = fp32_out.flatten()
static_flat = static_out.flatten()
correlation = np.corrcoef(fp32_flat, static_flat)[0, 1]
print(f"\nCorrelation between FP32 and Static outputs: {correlation:.4f}")
print("  (close to 1.0 = same pattern, different scale -> SCALING bug)")
print("  (close to 0.0 = genuinely different/broken -> CALIBRATION bug)")