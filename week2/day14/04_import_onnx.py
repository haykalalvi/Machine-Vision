"""
Day 14 - Script 4: Export both stages to ONNX
"""

from ultralytics import YOLO
import torch
import segmentation_models_pytorch as smp

YOLO_PATH = '/Users/alvi/OpenCV/week2/day14/runs/stage1_detector/weights/best.pt'
UNET_PATH = '/Users/alvi/OpenCV/output/best_unet.pt'

# --- Export Stage 1: YOLOv8 ---
print("Exporting Stage 1 (YOLOv8) to ONNX...")
yolo_model = YOLO(YOLO_PATH)
yolo_model.export(format='onnx', dynamic=True, simplify=True, imgsz=512)
print("  Saved: stage1_detector/weights/best.onnx")

# --- Export Stage 2: U-Net ---
print("\nExporting Stage 2 (U-Net) to ONNX...")
unet = smp.Unet(
    encoder_name='resnet34', encoder_weights=None,
    in_channels=3, classes=1, activation=None,
)
unet.load_state_dict(torch.load(UNET_PATH, map_location='cpu'))
unet.eval()

dummy_input = torch.randn(1, 3, 256, 256)
torch.onnx.export(
    unet, dummy_input,
    'output/unet_stage2.onnx',
    input_names=['input'],
    output_names=['mask_logits'],
    dynamic_axes={
        'input':       {0: 'batch_size'},
        'mask_logits': {0: 'batch_size'},
    },
    opset_version=17,
)
print("  Saved: output/unet_stage2.onnx")

print("\n✓ Both stages exported. Verify with Netron (netron.app):")
print("  - stage1_detector/weights/best.onnx")
print("  - output/unet_stage2.onnx")