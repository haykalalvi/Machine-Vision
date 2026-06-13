"""
Day 16 - Script 2: The 2x2 Deployment Matrix
Rows:    FP32 vs Static INT8 (Day 15's quantization variable)
Columns: ONNX Runtime vs OpenVINO (Day 16's runtime variable)

This isolates each variable independently -- the proper way to
benchmark when two optimizations could interact.
"""

import openvino as ov
import onnxruntime as ort
import numpy as np
import time
from pathlib import Path

ONNX_PATHS = {
    'FP32':        '/Users/alvi/OpenCV/week2/day14/runs/stage1_detector/weights/best.onnx',
    'Static INT8': '/Users/alvi/OpenCV/output/stage1_static_int8.onnx',
}
IR_PATHS = {
    'FP32':        'output/openvino_ir/stage1_fp32.xml',
    'Static INT8': 'output/openvino_ir/stage1_static_int8.xml',
}

IMG_SIZE = 512
N_WARMUP = 10
N_RUNS   = 50


def benchmark_onnxruntime(onnx_path, input_shape, n_warmup, n_runs):
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    dummy = np.random.randn(*input_shape).astype(np.float32)

    for _ in range(n_warmup):
        session.run(None, {input_name: dummy})

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy})
        times.append((time.perf_counter() - t0) * 1000)

    return np.array(times)


def benchmark_openvino(ir_path, input_shape, n_warmup, n_runs, device='CPU'):
    core = ov.Core()
    model = core.read_model(ir_path)
    compiled_model = core.compile_model(model, device)

    input_layer = compiled_model.input(0)
    dummy = np.random.randn(*input_shape).astype(np.float32)

    for _ in range(n_warmup):
        compiled_model([dummy])

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        compiled_model([dummy])
        times.append((time.perf_counter() - t0) * 1000)

    return np.array(times)


# ============================================================
# RUN THE FULL 2x2 MATRIX
# ============================================================
input_shape = (1, 3, IMG_SIZE, IMG_SIZE)
results = {}

print("Running 2x2 deployment matrix benchmark...")
print(f"Input shape: {input_shape}, warmup={N_WARMUP}, runs={N_RUNS}\n")

for precision in ['FP32', 'Static INT8']:
    # --- ONNX Runtime ---
    print(f"  [{precision}] ONNX Runtime...")
    ort_times = benchmark_onnxruntime(ONNX_PATHS[precision], input_shape, N_WARMUP, N_RUNS)
    results[(precision, 'ONNX Runtime')] = ort_times

    # --- OpenVINO ---
    print(f"  [{precision}] OpenVINO...")
    ov_times = benchmark_openvino(IR_PATHS[precision], input_shape, N_WARMUP, N_RUNS)
    results[(precision, 'OpenVINO')] = ov_times

# ============================================================
# RESULTS TABLE
# ============================================================
print(f"\n{'='*70}")
print("2x2 DEPLOYMENT MATRIX RESULTS")
print('='*70)
print(f"{'Precision':<14} {'Runtime':<14} {'Mean ms':>9} {'Std ms':>8} {'FPS':>8}")
print('-'*70)

for precision in ['FP32', 'Static INT8']:
    for runtime in ['ONNX Runtime', 'OpenVINO']:
        times = results[(precision, runtime)]
        mean_ms = times.mean()
        std_ms  = times.std()
        fps     = 1000 / mean_ms
        print(f"{precision:<14} {runtime:<14} {mean_ms:>8.2f} {std_ms:>8.2f} {fps:>8.1f}")

# ============================================================
# ISOLATE EACH OPTIMIZATION'S EFFECT
# ============================================================
fp32_ort  = results[('FP32', 'ONNX Runtime')].mean()
int8_ort  = results[('Static INT8', 'ONNX Runtime')].mean()
fp32_ov   = results[('FP32', 'OpenVINO')].mean()
int8_ov   = results[('Static INT8', 'OpenVINO')].mean()

print(f"\n{'='*70}")
print("ISOLATING EACH OPTIMIZATION (controlling for the other variable)")
print('='*70)

print(f"\nQuantization effect (FP32 -> Static INT8):")
print(f"  Within ONNX Runtime: {fp32_ort:.2f}ms -> {int8_ort:.2f}ms "
      f"({fp32_ort/int8_ort:.2f}x speedup)  <- this is Day 15's number")
print(f"  Within OpenVINO:     {fp32_ov:.2f}ms -> {int8_ov:.2f}ms "
      f"({fp32_ov/int8_ov:.2f}x speedup)")

print(f"\nRuntime effect (ONNX Runtime -> OpenVINO):")
print(f"  At FP32:        {fp32_ort:.2f}ms -> {fp32_ov:.2f}ms "
      f"({fp32_ort/fp32_ov:.2f}x speedup)")
print(f"  At Static INT8: {int8_ort:.2f}ms -> {int8_ov:.2f}ms "
      f"({int8_ort/int8_ov:.2f}x speedup)")

print(f"\nBest overall: ", end="")
best = min(results.items(), key=lambda x: x[1].mean())
print(f"{best[0][0]} + {best[0][1]} = {best[1].mean():.2f}ms "
      f"({1000/best[1].mean():.1f} FPS)")

np.save('output/matrix_results.npy', {k: v for k, v in results.items()})