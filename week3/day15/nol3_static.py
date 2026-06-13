"""
Day 15 - Script 3: Static Quantization with Calibration
Both weights AND activations quantized to INT8.
Calibration: run real leather images through the model to measure
the ACTUAL RANGE of activation values at each layer.

Why this matters: INT8 has only 256 levels (0-255). If an activation
naturally ranges from -50 to +50, mapping that to 0-255 preserves
detail. If we guessed a range of -1000 to +1000 instead, most real
values would cluster in a tiny corner of the INT8 range -- wasting
precision. Calibration measures the REAL range from REAL data.
"""

from onnxruntime.quantization import (
    quantize_static, QuantType, QuantFormat,
    CalibrationDataReader, CalibrationMethod
)
import numpy as np
import cv2
from pathlib import Path
import sys
sys.path.append('.')
from nol1_benchmark import benchmark_onnx_model, print_benchmark

FP32_PATH = '/Users/alvi/OpenCV/week2/day14/runs/stage1_detector/weights/best.onnx'
INT8_STATIC_PATH = 'output/stage1_static_int8.onnx'
DATA_ROOT = Path('/Users/alvi/OpenCV/data/leather')
IMG_SIZE = 512  # must match Day 14 training imgsz


class LeatherCalibrationReader(CalibrationDataReader):
    """
    Feeds real leather images to the quantizer for calibration.

    Uses a MIX of good + defective images -- the model needs to see
    the activation ranges it will actually encounter in production,
    which includes both normal AND defective surface textures.
    """
    def __init__(self, data_root, img_size=512, n_samples=100):
        self.img_size = img_size
        self.input_name = 'images'  # YOLOv8 ONNX input name

        # Collect images from ALL categories (good + each defect type)
        # for representative calibration
        all_paths = []
        test_path = data_root / 'test'
        for category_dir in sorted(test_path.iterdir()):
            if category_dir.is_dir():
                all_paths.extend(sorted(category_dir.glob('*.png')))

        # Shuffle and take n_samples
        rng = np.random.RandomState(42)
        rng.shuffle(all_paths)
        self.image_paths = all_paths[:n_samples]

        print(f"  Calibration set: {len(self.image_paths)} images "
              f"(mixed good + defective categories)")

        self.enum_data = None

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = self._preprocess_all()
        return next(self.enum_data, None)

    def _preprocess_all(self):
        for img_path in self.image_paths:
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.img_size, self.img_size))

            # YOLOv8 preprocessing: normalize to [0,1], CHW format
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
            img = np.expand_dims(img, axis=0)   # add batch dim

            yield {self.input_name: img}

    def rewind(self):
        self.enum_data = None


# ============================================================
# DETERMINE INPUT NAME FIRST
# ============================================================
import onnx
model = onnx.load(FP32_PATH)
input_name = model.graph.input[0].name
print(f"Model input name: {input_name}")

calibration_reader = LeatherCalibrationReader(
    DATA_ROOT, img_size=IMG_SIZE, n_samples=100
)
calibration_reader.input_name = input_name

# ============================================================
# RUN STATIC QUANTIZATION
# ============================================================
print("\nRunning static quantization with calibration...")
print("This measures real activation ranges from leather images...")

# quantize_static(
#     model_input=FP32_PATH,
#     model_output=INT8_STATIC_PATH,
#     calibration_data_reader=calibration_reader,
#     quant_format=QuantFormat.QDQ,        # Quantize-Dequantize format
#                                           # -- best CPU compatibility
#     weight_type=QuantType.QInt8,
#     activation_type=QuantType.QInt8,
#     calibrate_method=CalibrationMethod.MinMax,
#     # MinMax: simplest calibration -- track observed min/max per tensor
#     # Alternative: CalibrationMethod.Entropy (more accurate, slower)
# )

# Add to your quantize_static() call in Script 3:
quantize_static(
    model_input=FP32_PATH,
    model_output=INT8_STATIC_PATH,
    calibration_data_reader=calibration_reader,
    quant_format=QuantFormat.QDQ,
    weight_type=QuantType.QInt8,
    activation_type=QuantType.QInt8,
    calibrate_method=CalibrationMethod.MinMax,

    # NEW: only quantize Conv and MatMul ops -- leave detection
    # head's final activation/concat operations in FP32
    op_types_to_quantize=['Conv', 'MatMul'],
)

print(f"\nSaved: {INT8_STATIC_PATH}")

# ============================================================
# BENCHMARK
# ============================================================
print("\nBenchmarking statically quantized model...")
static_stats = benchmark_onnx_model(
    INT8_STATIC_PATH,
    input_shape=(1, 3, IMG_SIZE, IMG_SIZE),
    n_warmup=10,
    n_runs=50,
)
print_benchmark("Static INT8 (Stage 1 YOLOv8)", static_stats)

# ============================================================
# THREE-WAY COMPARISON
# ============================================================
fp32_stats   = np.load('output/fp32_baseline.npy', allow_pickle=True).item()
dynamic_stats = np.load('output/int8_dynamic.npy', allow_pickle=True).item()

print(f"\n{'='*55}")
print("THREE-WAY COMPARISON")
print('='*55)
print(f"{'Model':<20} {'Latency':>10} {'FPS':>8} {'Size':>10} {'Speedup':>9}")
print('-'*60)
print(f"{'FP32 (baseline)':<20} {fp32_stats['mean_ms']:>9.2f}ms "
      f"{fp32_stats['fps']:>7.1f} {fp32_stats['model_size_mb']:>8.2f}MB "
      f"{'1.00x':>9}")
print(f"{'Dynamic INT8':<20} {dynamic_stats['mean_ms']:>9.2f}ms "
      f"{dynamic_stats['fps']:>7.1f} {dynamic_stats['model_size_mb']:>8.2f}MB "
      f"{fp32_stats['mean_ms']/dynamic_stats['mean_ms']:>8.2f}x")
print(f"{'Static INT8':<20} {static_stats['mean_ms']:>9.2f}ms "
      f"{static_stats['fps']:>7.1f} {static_stats['model_size_mb']:>8.2f}MB "
      f"{fp32_stats['mean_ms']/static_stats['mean_ms']:>8.2f}x")

np.save('output/int8_static.npy', static_stats)