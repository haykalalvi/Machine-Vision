"""
Day 16 - Diagnostic v2: Compare ONNX Runtime vs OpenVINO output SHAPES
Fixed: handle dynamic shapes (use actual array shapes after inference,
not get_shape() on the declared model output)
Fixed: wrap in __name__ guard to prevent macOS multiprocessing re-exec
"""

import openvino as ov
import onnxruntime as ort
import numpy as np
import cv2
from pathlib import Path


def main():
    DATA_ROOT = Path('/Users/alvi/OpenCV/data/leather')
    IMG_SIZE = 512

    img_path = sorted((DATA_ROOT / 'test' / 'cut').glob('*.png'))[0]
    img = cv2.imread(str(img_path))
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_chw = np.transpose(img_norm, (2, 0, 1))
    input_tensor = np.expand_dims(img_chw, axis=0)

    print(f"Input tensor shape: {input_tensor.shape}")

    # ============================================================
    # ONNX RUNTIME
    # ============================================================
    # ort_session = ort.InferenceSession(
    #     '/Users/alvi/OpenCV/output/stage1_static_int8.onnx',
    #     providers=['CPUExecutionProvider']
    # )
    ort_session = ort.InferenceSession(
    '/Users/alvi/OpenCV/week2/day14/runs/stage1_detector/weights/best.onnx',  # FP32
    providers=['CPUExecutionProvider']
)

    print("\n" + "=" * 55)
    print("ONNX RUNTIME -- declared outputs (may show dynamic dims)")
    print("=" * 55)
    for i, out in enumerate(ort_session.get_outputs()):
        print(f"  Output {i}: name='{out.name}', declared_shape={out.shape}")

    ort_results = ort_session.run(None, {'images': input_tensor})
    print(f"\nONNX Runtime -- ACTUAL output after inference:")
    for i, r in enumerate(ort_results):
        print(f"  ort_results[{i}].shape = {r.shape}, "
              f"range=[{r.min():.3f}, {r.max():.3f}]")

    # ============================================================
    # OPENVINO -- skip get_shape() on declared outputs, go straight
    # to running inference and checking the ACTUAL result shape
    # ============================================================
    core = ov.Core()
    # ov_model = core.read_model('output/openvino_ir/stage1_static_int8.xml')
    ov_model = core.read_model('output/openvino_ir/stage1_fp32.xml')  # FP32
    ov_compiled = core.compile_model(ov_model, 'CPU')

    print("\n" + "=" * 55)
    print("OPENVINO -- declared outputs (partial/dynamic shapes)")
    print("=" * 55)
    for i, out in enumerate(ov_compiled.outputs):
        print(f"  Output {i}: names={out.get_names()}, "
              f"partial_shape={out.get_partial_shape()}")

    # Run inference using EXPLICIT named input/output
    input_key  = ov_compiled.input(0)
    output_key = ov_compiled.output(0)
    print(f"\n  Input key:  {input_key.get_any_name()}")
    print(f"  Output key: {output_key.get_any_name()}")

    ov_result = ov_compiled({input_key: input_tensor})
    ov_arr = ov_result[output_key]

    print(f"\nOpenVINO -- ACTUAL output after inference:")
    print(f"  ov_arr.shape = {ov_arr.shape}, "
          f"range=[{ov_arr.min():.3f}, {ov_arr.max():.3f}]")

    # ============================================================
    # COMPARE
    # ============================================================
    print("\n" + "=" * 55)
    print("COMPARISON")
    print("=" * 55)
    print(f"  ONNX shape:     {ort_results[0].shape}")
    print(f"  OpenVINO shape: {ov_arr.shape}")

    if ort_results[0].shape != ov_arr.shape:
        print("\n  *** SHAPE MISMATCH FOUND ***")
        print("  This IS your IoU=0.74 root cause.")
        print(f"  ONNX anchors:     {ort_results[0].shape[-1]}")
        print(f"  OpenVINO anchors: {ov_arr.shape[-1]}")
        print("\n  Likely cause: OpenVINO's compiled model resolved the")
        print("  dynamic input/output shape differently than expected")
        print("  for a 512x512 input -- check if it's using a DEFAULT")
        print("  shape (e.g. 640x640 -> 8400 anchors) instead of 512x512.")
    else:
        diff = np.abs(ort_results[0] - ov_arr)
        correlation = np.corrcoef(ort_results[0].flatten(), ov_arr.flatten())[0, 1]
        print(f"\n  Shapes MATCH")
        print(f"  Max abs diff:  {diff.max():.6f}")
        print(f"  Mean abs diff: {diff.mean():.6f}")
        print(f"  Correlation:   {correlation:.6f}")


if __name__ == '__main__':
    main()