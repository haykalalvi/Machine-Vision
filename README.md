# Machine Vision Engineering Curriculum
Self-directed deep learning curriculum covering classical CV, deep
learning architectures, and production deployment, applied to
manufacturing quality-control use cases.

## Highlighted project
**[Two-Stage Defect Detection + Edge Deployment Optimization](week3/day14-16/)**
YOLOv8 + U-Net pipeline, ONNX Runtime / OpenVINO benchmarking,
67 FPS via INT8 quantization (3.38x speedup, IoU>0.90).

# Results at a glance

| Configuration              | Latency  | FPS  | IoU vs FP32 |
|-----------------------------|----------|------|-------------|
| FP32, ONNX Runtime (baseline) | 50.40ms | 19.8 | 1.000 |
| Static INT8, ONNX Runtime    | 14.93ms | 67.0 | >0.90 |
| FP32, OpenVINO               | 15.94ms | 62.7 | 1.000 |
| Static INT8, OpenVINO        | 19.51ms | 51.2 | >0.90 |

**Best configuration: Static INT8 + ONNX Runtime — 3.38x speedup, 67 FPS, IoU>0.90**

OpenVINO improves the unoptimized FP32 model by 3.16x (50.40ms → 15.94ms).
But applying OpenVINO on TOP of an already-quantized INT8 model makes it
SLOWER (14.93ms → 19.51ms, a 0.76x "speedup").

Why: both ONNX Runtime's static INT8 quantization and OpenVINO's graph
optimization perform overlapping operator fusion. Once quantization has
already removed redundant computation, OpenVINO's conversion step adds
format, translation overhead with no remaining fusion opportunities to
exploit.

## Structure
| Week | Focus | Key topics |
|------|-------|-----------|
| 1 | Classical CV foundations | Camera calibration, homography, classical defect detection |
| 2 | Deep learning for vision | CNNs, EfficientNet-B0 transfer learning, YOLOv8, U-Net, PatchCore |
| 3 | Production engineering | ONNX, OpenVINO quantization, deployment benchmarking |

## Tech stack
OpenCV, PyTorch, Ultralytics YOLOv8, segmentation-models-pytorch,
ONNX Runtime, OpenVINO, Albumentations, Weights & Biases
