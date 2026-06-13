"""
Day 16 - Script 1: Convert ONNX models to OpenVINO IR format
Converts BOTH Day 15 models (FP32 and Static INT8) so we can
build the full 2x2 comparison matrix.
"""

import openvino as ov
from pathlib import Path

ONNX_PATHS = {
    'FP32':        '/Users/alvi/OpenCV/week2/day14/runs/stage1_detector/weights/best.onnx',
    'Static INT8': '/Users/alvi/OpenCV/output/stage1_static_int8.onnx',
}

OUTPUT_DIR = Path('output/openvino_ir')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"OpenVINO version: {ov.__version__}")
# print(f"OpenVINO version: {ov.get_version()}")

# print(f"OpenVINO version: {ov.runtime.get_version()}")

# print(f"Loading penVINO from: {ov.__file__}")
# ============================================================
# CHECK AVAILABLE DEVICES ON THIS MACHINE
# ============================================================
core = ov.Core()
available_devices = core.available_devices
print(f"\nAvailable OpenVINO devices on this machine: {available_devices}")
print("  (On Apple Silicon, expect: ['CPU'] only -- no Intel GPU/NPU)")
print("  (On Intel hardware, you might see: ['CPU', 'GPU', 'NPU'])")

for device in available_devices:
    full_name = core.get_property(device, "FULL_DEVICE_NAME")
    print(f"  {device}: {full_name}")

# ============================================================
# CONVERT EACH MODEL TO IR
# ============================================================
ir_paths = {}

for name, onnx_path in ONNX_PATHS.items():
    print(f"\n{'='*55}")
    print(f"Converting: {name}")
    print('='*55)

    # ov.convert_model reads the ONNX graph and applies
    # Intel-specific graph optimizations during conversion
    ov_model = ov.convert_model(onnx_path)

    # Save as IR (.xml + .bin pair)
    safe_name = name.replace(' ', '_').lower()
    ir_path = OUTPUT_DIR / f'stage1_{safe_name}.xml'
    ov.save_model(ov_model, str(ir_path))

    bin_path = ir_path.with_suffix('.bin')
    xml_size = ir_path.stat().st_size / 1024
    bin_size = bin_path.stat().st_size / (1024*1024)

    print(f"  Saved: {ir_path.name} ({xml_size:.1f} KB -- graph structure)")
    print(f"  Saved: {bin_path.name} ({bin_size:.2f} MB -- weights)")

    ir_paths[name] = str(ir_path)

print(f"\n{'='*55}")
print("CONVERSION COMPLETE")
print('='*55)
for name, path in ir_paths.items():
    print(f"  {name}: {path}")

print("\nNext: Script 2 benchmarks these against your Day 15 ONNX Runtime numbers")