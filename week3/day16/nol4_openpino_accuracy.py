"""
Day 16 - Script 4: Validate OpenVINO IR produces same detections as ONNX
Reuses Day 15's IoU comparison methodology, but compares:
  ONNX Runtime (Static INT8) vs OpenVINO IR (Static INT8)

If IR conversion is a faithful graph translation, IoU should be
very close to 1.0 -- this is verifying a FORMAT CONVERSION, not
a numerical approximation like quantization was.
"""

import openvino as ov
import onnxruntime as ort
import numpy as np
import cv2
from pathlib import Path
import yaml

DATA_ROOT = Path('/Users/alvi/OpenCV/data/leather')
DATA_YAML = '/Users/alvi/OpenCV/week2/day14/data/leather_yolo/data.yaml'
IMG_SIZE = 512
CONF_THRESH = 0.25
IOU_THRESH = 0.45

with open(DATA_YAML) as f:
    config = yaml.safe_load(f)
CLASS_NAMES = config['names']


def preprocess(img, img_size=512):
    img_resized = cv2.resize(img, (img_size, img_size))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_chw = np.transpose(img_norm, (2, 0, 1))
    return np.expand_dims(img_chw, axis=0), (img.shape[1], img.shape[0])


def postprocess(output, orig_size, img_size=512, conf_thresh=0.25, iou_thresh=0.45):
    predictions = output[0][0].T
    boxes_xywh = predictions[:, :4]
    scores = predictions[:, 4:]
    class_ids = np.argmax(scores, axis=1)
    confidences = np.max(scores, axis=1)

    mask = confidences > conf_thresh
    boxes_xywh, confidences, class_ids = boxes_xywh[mask], confidences[mask], class_ids[mask]
    if len(boxes_xywh) == 0:
        return []

    boxes_xyxy = np.zeros_like(boxes_xywh)
    boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2

    scale_x, scale_y = orig_size[0]/img_size, orig_size[1]/img_size
    boxes_xyxy[:, [0,2]] *= scale_x
    boxes_xyxy[:, [1,3]] *= scale_y

    indices = cv2.dnn.NMSBoxes(boxes_xyxy.tolist(), confidences.tolist(), conf_thresh, iou_thresh)
    if len(indices) == 0:
        return []
    indices = indices.flatten()

    return [{'bbox': boxes_xyxy[i].astype(int).tolist(),
             'confidence': float(confidences[i]),
             'class_id': int(class_ids[i])} for i in indices]


def compute_iou(box1, box2):
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = area1 + area2 - inter
    return inter/union if union > 0 else 0


# ============================================================
# LOAD BOTH RUNTIMES (same precision: Static INT8)
# ============================================================
ort_session = ort.InferenceSession(
    '/Users/alvi/OpenCV/output/stage1_static_int8.onnx',
    providers=['CPUExecutionProvider']
)

core = ov.Core()
ov_model = core.read_model('output/openvino_ir/stage1_static_int8.xml')
ov_compiled = core.compile_model(ov_model, 'CPU')

# ============================================================
# RUN ON TEST IMAGES
# ============================================================
test_path = DATA_ROOT / 'test'
test_images = []
for category_dir in sorted(test_path.iterdir()):
    if category_dir.is_dir():
        test_images.extend(sorted(category_dir.glob('*.png'))[:8])

print(f"Validating ONNX Runtime vs OpenVINO IR on {len(test_images)} images...")

ious = []
count_diffs = []

for img_path in test_images:
    img = cv2.imread(str(img_path))
    input_tensor, orig_size = preprocess(img, IMG_SIZE)

    # ONNX Runtime
    ort_out = ort_session.run(None, {'images': input_tensor})
    ort_dets = postprocess(ort_out, orig_size, IMG_SIZE, CONF_THRESH, IOU_THRESH)

    # OpenVINO
    ov_out = ov_compiled([input_tensor])
    ov_out_arr = [ov_out[ov_compiled.output(0)]]
    ov_dets = postprocess(ov_out_arr, orig_size, IMG_SIZE, CONF_THRESH, IOU_THRESH)

    count_diffs.append(len(ov_dets) - len(ort_dets))

    if ort_dets and ov_dets:
        per_box_ious = []
        for ort_det in ort_dets:
            best_iou = max([compute_iou(ort_det['bbox'], ov_det['bbox'])
                            for ov_det in ov_dets], default=0.0)
            per_box_ious.append(best_iou)
        ious.append(np.mean(per_box_ious))
    elif not ort_dets and not ov_dets:
        ious.append(1.0)
    else:
        ious.append(0.0)

# ============================================================
# RESULTS
# ============================================================
mean_iou = np.mean(ious)
identical = sum(1 for d in count_diffs if d == 0)

print(f"\n{'='*55}")
print("ONNX RUNTIME vs OPENVINO IR -- ACCURACY CHECK")
print('='*55)
print(f"Mean IoU (ONNX Runtime vs OpenVINO): {mean_iou:.4f}")
print(f"Images with identical detection count: {identical}/{len(test_images)}")

if mean_iou > 0.99:
    print("\nVERDICT: IR conversion is numerically equivalent -- expected,")
    print("since this is a graph FORMAT conversion, not a numerical")
    print("approximation. Any speed difference comes purely from the")
    print("runtime's execution engine, not from changed predictions.")
elif mean_iou > 0.95:
    print("\nVERDICT: Near-identical, tiny differences likely from")
    print("floating-point operation ordering differences between engines.")
else:
    print("\nVERDICT: Unexpected divergence -- investigate further")
    print("(this would be unusual for a same-precision format conversion)")

np.save('output/openvino_accuracy.npy', {'mean_iou': mean_iou, 'count_diffs': count_diffs})