"""
Day 15 - Script 4: Accuracy Validation After Quantization
Quantization can silently degrade accuracy. We must verify detections
are still correct -- not just measure speed.

Method: run all 3 model versions (FP32, dynamic INT8, static INT8) on
the same test images. Compare detection counts, confidence scores,
and bounding box positions (IoU between FP32 and quantized boxes).
"""

import onnxruntime as ort
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import yaml

DATA_ROOT = Path('/Users/alvi/OpenCV/data/leather')
DATA_YAML = '/Users/alvi/OpenCV/week2/day14/data/leather_yolo/data.yaml'
IMG_SIZE  = 512
CONF_THRESH = 0.25
IOU_THRESH  = 0.45

with open(DATA_YAML) as f:
    config = yaml.safe_load(f)
CLASS_NAMES = config['names']

MODEL_PATHS = {
    'FP32':       '/Users/alvi/OpenCV/week2/day14/runs/stage1_detector/weights/best.onnx',
    'Dynamic INT8': 'output/stage1_dynamic_int8.onnx',
    'Static INT8':  'output/stage1_static_int8.onnx',
}


def preprocess(img, img_size=512):
    """YOLOv8 ONNX preprocessing"""
    h, w = img.shape[:2]
    img_resized = cv2.resize(img, (img_size, img_size))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_chw  = np.transpose(img_norm, (2, 0, 1))
    img_batch = np.expand_dims(img_chw, axis=0)
    return img_batch, (w, h)


def postprocess(output, orig_size, img_size=512,
                conf_thresh=0.25, iou_thresh=0.45):
    """
    Decode YOLOv8 ONNX output -> bounding boxes + class + confidence
    YOLOv8 ONNX output shape: [1, 4+num_classes, num_anchors]
    """
    predictions = output[0]  # [1, 4+nc, num_anchors]
    predictions = predictions[0].T  # [num_anchors, 4+nc]

    boxes_xywh = predictions[:, :4]
    scores     = predictions[:, 4:]

    class_ids = np.argmax(scores, axis=1)
    confidences = np.max(scores, axis=1)

    mask = confidences > conf_thresh
    boxes_xywh = boxes_xywh[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]

    if len(boxes_xywh) == 0:
        return []

    # Convert xywh (center) -> xyxy
    boxes_xyxy = np.zeros_like(boxes_xywh)
    boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2

    # Scale from img_size back to original image size
    scale_x = orig_size[0] / img_size
    scale_y = orig_size[1] / img_size
    boxes_xyxy[:, [0,2]] *= scale_x
    boxes_xyxy[:, [1,3]] *= scale_y

    # NMS
    indices = cv2.dnn.NMSBoxes(
        boxes_xyxy.tolist(), confidences.tolist(), conf_thresh, iou_thresh
    )
    if len(indices) == 0:
        return []
    indices = indices.flatten()

    results = []
    for i in indices:
        results.append({
            'bbox': boxes_xyxy[i].astype(int).tolist(),
            'confidence': float(confidences[i]),
            'class_id': int(class_ids[i]),
        })
    return results


def compute_iou(box1, box2):
    """IoU between two [x1,y1,x2,y2] boxes"""
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = area1 + area2 - inter
    return inter/union if union > 0 else 0


# ============================================================
# LOAD ALL THREE SESSIONS
# ============================================================
sessions = {}
for name, path in MODEL_PATHS.items():
    sessions[name] = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    print(f"Loaded: {name}")

# ============================================================
# COLLECT TEST IMAGES -- across all defect types + good
# ============================================================
test_path = DATA_ROOT / 'test'
test_images = []
for category_dir in sorted(test_path.iterdir()):
    if category_dir.is_dir():
        imgs = sorted(category_dir.glob('*.png'))[:8]  # 8 per category
        test_images.extend([(p, category_dir.name) for p in imgs])

print(f"\nValidating on {len(test_images)} images across "
      f"{len(set(c for _,c in test_images))} categories...")

# ============================================================
# RUN ALL MODELS ON ALL IMAGES
# ============================================================
results_per_model = {name: [] for name in MODEL_PATHS}
iou_vs_fp32 = {'Dynamic INT8': [], 'Static INT8': []}
detection_count_diff = {'Dynamic INT8': [], 'Static INT8': []}

for img_path, category in test_images:
    img = cv2.imread(str(img_path))
    input_tensor, orig_size = preprocess(img, IMG_SIZE)

    detections_by_model = {}
    for name, sess in sessions.items():
        output = sess.run(None, {'images': input_tensor})
        dets = postprocess(output, orig_size, IMG_SIZE, CONF_THRESH, IOU_THRESH)
        detections_by_model[name] = dets
        results_per_model[name].append({
            'category': category,
            'n_detections': len(dets),
            'max_conf': max([d['confidence'] for d in dets], default=0.0),
        })

    # Compare INT8 versions against FP32
    fp32_dets = detections_by_model['FP32']
    for int8_name in ['Dynamic INT8', 'Static INT8']:
        int8_dets = detections_by_model[int8_name]
        detection_count_diff[int8_name].append(
            len(int8_dets) - len(fp32_dets)
        )

        # For each FP32 detection, find best-matching IoU in INT8 detections
        if fp32_dets and int8_dets:
            ious = []
            for fp32_det in fp32_dets:
                best_iou = max(
                    [compute_iou(fp32_det['bbox'], int8_det['bbox'])
                     for int8_det in int8_dets],
                    default=0.0
                )
                ious.append(best_iou)
            iou_vs_fp32[int8_name].append(np.mean(ious))
        elif not fp32_dets and not int8_dets:
            iou_vs_fp32[int8_name].append(1.0)  # both correctly found nothing
        else:
            iou_vs_fp32[int8_name].append(0.0)  # one found something, other didn't

# ============================================================
# SUMMARY STATISTICS
# ============================================================
print(f"\n{'='*60}")
print("ACCURACY VALIDATION SUMMARY")
print('='*60)

for name in MODEL_PATHS:
    total_dets = sum(r['n_detections'] for r in results_per_model[name])
    avg_conf   = np.mean([r['max_conf'] for r in results_per_model[name] if r['max_conf']>0])
    print(f"\n{name}:")
    print(f"  Total detections across {len(test_images)} images: {total_dets}")
    print(f"  Avg confidence (when detected): {avg_conf:.3f}")

print(f"\n{'='*60}")
print("QUANTIZATION IMPACT (vs FP32)")
print('='*60)
for int8_name in ['Dynamic INT8', 'Static INT8']:
    mean_iou = np.mean(iou_vs_fp32[int8_name])
    diff_counts = detection_count_diff[int8_name]
    print(f"\n{int8_name}:")
    print(f"  Mean IoU vs FP32 boxes:     {mean_iou:.4f}  "
          f"({'GOOD' if mean_iou>0.9 else 'CHECK' if mean_iou>0.7 else 'BAD'})")
    print(f"  Detection count changes:    "
          f"{sum(1 for d in diff_counts if d==0)}/{len(diff_counts)} images identical")
    print(f"  Images with MORE detections: {sum(1 for d in diff_counts if d>0)}")
    print(f"  Images with FEWER detections: {sum(1 for d in diff_counts if d<0)}")

# ============================================================
# VISUALIZATION
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Quantization Accuracy Impact (vs FP32 baseline)', fontsize=13, fontweight='bold')

# IoU distribution
for int8_name, color in [('Dynamic INT8', '#378ADD'), ('Static INT8', '#1D9E75')]:
    axes[0].hist(iou_vs_fp32[int8_name], bins=20, alpha=0.6,
                 label=int8_name, color=color)
axes[0].axvline(0.9, color='red', linestyle='--', label='0.90 threshold')
axes[0].set_title('Box IoU vs FP32\n(1.0 = identical boxes)')
axes[0].set_xlabel('IoU'); axes[0].set_ylabel('Count')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

# Detection count differences
for int8_name, color in [('Dynamic INT8', '#378ADD'), ('Static INT8', '#1D9E75')]:
    axes[1].hist(detection_count_diff[int8_name], bins=range(-3,4), alpha=0.6,
                 label=int8_name, color=color, align='left')
axes[1].axvline(0, color='black', linestyle='-', linewidth=1)
axes[1].set_title('Detection count change\n(0 = same as FP32)')
axes[1].set_xlabel('Δ detections vs FP32')
axes[1].legend(); axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('output/04_accuracy_validation.jpg', dpi=150)
plt.show()
print("\nSaved: output/04_accuracy_validation.jpg")