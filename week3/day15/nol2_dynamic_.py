"""
Day 15 - Script 2: Dynamic Quantization
Converts FP32 weights -> INT8. Activations quantized on-the-fly.
No calibration data needed -- fastest path to a speedup.
"""

from onnxruntime.quantization import quantize_dynamic, QuantType
from pathlib import Path
import sys
sys.path.append('.')
from nol1_benchmark import benchmark_onnx_model, print_benchmark

FP32_PATH = '/Users/alvi/OpenCV/week2/day14/runs/stage1_detector/weights/best.onnx'
INT8_DYNAMIC_PATH = 'output/stage1_dynamic_int8.onnx'

print("Applying dynamic quantization...")
print("  Weights -> INT8 (offline)")
print("  Activations -> INT8 (computed on-the-fly per inference)")

quantize_dynamic(
    model_input=FP32_PATH,
    model_output=INT8_DYNAMIC_PATH,
    weight_type=QuantType.QInt8,
)

print(f"\nSaved: {INT8_DYNAMIC_PATH}")

# --- Benchmark ---
print("\nBenchmarking quantized model...")
int8_stats = benchmark_onnx_model(
    INT8_DYNAMIC_PATH,
    input_shape=(1, 3, 512, 512),
    n_warmup=10,
    n_runs=50,
)
print_benchmark("Dynamic INT8 (Stage 1 YOLOv8)", int8_stats)

# --- Compare ---
import numpy as np
fp32_stats = np.load('output/fp32_baseline.npy', allow_pickle=True).item()

speedup = fp32_stats['mean_ms'] / int8_stats['mean_ms']
size_reduction = fp32_stats['model_size_mb'] / int8_stats['model_size_mb']

print(f"\n{'='*55}")
print("DYNAMIC QUANTIZATION RESULTS")
print('='*55)
print(f"  Latency:  {fp32_stats['mean_ms']:.2f}ms -> {int8_stats['mean_ms']:.2f}ms "
      f"({speedup:.2f}x speedup)")
print(f"  FPS:      {fp32_stats['fps']:.1f} -> {int8_stats['fps']:.1f}")
print(f"  Size:     {fp32_stats['model_size_mb']:.2f}MB -> {int8_stats['model_size_mb']:.2f}MB "
      f"({size_reduction:.2f}x smaller)")

np.save('output/int8_dynamic.npy', int8_stats)