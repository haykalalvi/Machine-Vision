"""
Day 15 - Script 1: Correct Benchmark Harness
Fixes the Day 14 warm-up artifact (340ms cold-start vs 19ms steady-state).

Methodology:
  1. Run N warm-up inferences (discarded) -- let backend compile/cache kernels
  2. Run M timed inferences -- measure these only
  3. Report mean, median, std, min, max -- not just one number
"""

import onnxruntime as ort
import numpy as np
import time
from pathlib import Path
import cv2

def benchmark_onnx_model(onnx_path, input_shape=(1, 3, 512, 512),
                          n_warmup=10, n_runs=50, providers=None):
    """
    Benchmark an ONNX model with proper warm-up separation.

    Returns dict with timing statistics in milliseconds.
    """
    if providers is None:
        providers = ['CPUExecutionProvider']

    session = ort.InferenceSession(onnx_path, providers=providers)
    input_name = session.get_inputs()[0].name

    print(f"  Model: {Path(onnx_path).name}")
    print(f"  Providers: {session.get_providers()}")
    print(f"  Input: {input_name} {input_shape}")

    dummy = np.random.randn(*input_shape).astype(np.float32)

    # --- WARM-UP (discarded) ---
    print(f"  Warming up ({n_warmup} runs)...")
    for _ in range(n_warmup):
        session.run(None, {input_name: dummy})

    # --- TIMED RUNS ---
    print(f"  Benchmarking ({n_runs} runs)...")
    times_ms = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy})
        times_ms.append((time.perf_counter() - t0) * 1000)

    times_ms = np.array(times_ms)

    return {
        'mean_ms':   float(np.mean(times_ms)),
        'median_ms': float(np.median(times_ms)),
        'std_ms':    float(np.std(times_ms)),
        'min_ms':    float(np.min(times_ms)),
        'max_ms':    float(np.max(times_ms)),
        'fps':       float(1000 / np.mean(times_ms)),
        'model_size_mb': Path(onnx_path).stat().st_size / (1024*1024),
        'raw_times': times_ms,
    }


def print_benchmark(name, stats):
    print(f"\n{'='*55}")
    print(f"{name}")
    print('='*55)
    print(f"  Mean:      {stats['mean_ms']:.2f} ms")
    print(f"  Median:    {stats['median_ms']:.2f} ms")
    print(f"  Std dev:   {stats['std_ms']:.2f} ms")
    print(f"  Min/Max:   {stats['min_ms']:.2f} / {stats['max_ms']:.2f} ms")
    print(f"  FPS:       {stats['fps']:.1f}")
    print(f"  Model size: {stats['model_size_mb']:.2f} MB")


if __name__ == '__main__':
    FP32_PATH = '/Users/alvi/OpenCV/week2/day14/runs/stage1_detector/weights/best.onnx'

    print("Benchmarking ORIGINAL (FP32) Stage 1 YOLOv8...")
    print("Note: input_shape must match your Day 14 training imgsz (512)")

    fp32_stats = benchmark_onnx_model(
        FP32_PATH,
        input_shape=(1, 3, 512, 512),
        n_warmup=10,
        n_runs=50,
    )
    print_benchmark("FP32 Baseline (Stage 1 YOLOv8)", fp32_stats)

    print(f"\n--- Comparison with Day 14 ---")
    print(f"Day 14 (with cold-start mixed in): 50.40ms avg")
    print(f"Day 15 (properly warmed up):       {fp32_stats['mean_ms']:.2f}ms avg")
    print(f"\nThis is your TRUE baseline for today's quantization comparisons.")

    np.save('output/fp32_baseline.npy', fp32_stats)