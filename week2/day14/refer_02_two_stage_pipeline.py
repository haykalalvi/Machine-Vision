"""
Day 14 - Script 2: Two-Stage Detection + Segmentation Pipeline

Architecture:
  Stage 1 (YOLOv8): full image -> bounding boxes (WHERE)
  Stage 2 (U-Net):  cropped region -> precise mask (WHAT EXACTLY)

This class is the deliverable artifact for your portfolio.
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import albumentations as A
from albumentations.pytorch import ToTensorV2
import time


class TwoStageDefectPipeline:
    """
    Stage 1 (YOLOv8) finds candidate defect regions.
    Stage 2 (U-Net) segments precise pixel masks within each region.

    Usage:
        pipeline = TwoStageDefectPipeline(yolo_path, unet_path)
        result = pipeline.predict(image)
    """

    def __init__(self, yolo_path, unet_path, device=None,
                 crop_padding=0.2, det_conf=0.25, seg_threshold=0.5,
                 unet_img_size=256):
        self.device = device or (
            'mps' if torch.backends.mps.is_available() else
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        # --- Load Stage 1: YOLOv8 ---
        self.detector = YOLO(yolo_path)
        self.det_conf = det_conf
        self.crop_padding = crop_padding
        print(f"Stage 1 (YOLOv8) loaded: {yolo_path}")
        print(f"  Classes: {self.detector.names}")

        # --- Load Stage 2: U-Net ---
        self.segmenter = smp.Unet(
            encoder_name='resnet34', encoder_weights=None,
            in_channels=3, classes=1, activation=None,
        )
        self.segmenter.load_state_dict(
            torch.load(unet_path, map_location=self.device)
        )
        self.segmenter = self.segmenter.to(self.device)
        self.segmenter.eval()
        self.seg_threshold = seg_threshold
        self.unet_img_size = unet_img_size
        print(f"Stage 2 (U-Net) loaded: {unet_path}")

        # U-Net preprocessing — must match Day 12 training transforms
        self.seg_transform = A.Compose([
            A.Resize(unet_img_size, unet_img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    def _expand_box(self, x1, y1, x2, y2, img_w, img_h):
        """Add padding around a bounding box, clamped to image bounds."""
        bw, bh = x2 - x1, y2 - y1
        pad_x  = bw * self.crop_padding
        pad_y  = bh * self.crop_padding

        x1 = max(0, int(x1 - pad_x))
        y1 = max(0, int(y1 - pad_y))
        x2 = min(img_w,  int(x2 + pad_x))
        y2 = min(img_h,  int(y2 + pad_y))
        return x1, y1, x2, y2

    @torch.no_grad()
    def _segment_crop(self, crop_bgr):
        """Run U-Net on a single cropped region. Returns mask at crop's original size."""
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = crop_rgb.shape[:2]

        transformed = self.seg_transform(image=crop_rgb)
        tensor = transformed['image'].unsqueeze(0).to(self.device)

        logits = self.segmenter(tensor)
        prob   = torch.sigmoid(logits)[0, 0].cpu().numpy()

        # Resize mask back to original crop dimensions
        mask = cv2.resize(prob, (orig_w, orig_h))
        return mask

    def predict(self, image_bgr):
        """
        Run the full two-stage pipeline on one image.

        Returns a dict:
          detections: list of {class, confidence, bbox, mask, defect_area_px,
                                defect_area_pct, mask_full}
          timing: {detect_ms, segment_ms, total_ms}
          full_mask: combined binary mask at original image resolution
        """
        h, w = image_bgr.shape[:2]
        full_mask = np.zeros((h, w), dtype=np.uint8)

        # --- STAGE 1: DETECT ---
        t0 = time.perf_counter()
        results = self.detector(image_bgr, conf=self.det_conf, verbose=False)[0]
        detect_ms = (time.perf_counter() - t0) * 1000

        detections = []
        segment_ms_total = 0

        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu())
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                cls_name = self.detector.names[cls_id]

                # Expand crop with padding
                ex1, ey1, ex2, ey2 = self._expand_box(x1, y1, x2, y2, w, h)
                crop = image_bgr[ey1:ey2, ex1:ex2]

                if crop.size == 0:
                    continue

                # --- STAGE 2: SEGMENT within crop ---
                t1 = time.perf_counter()
                mask_prob = self._segment_crop(crop)
                segment_ms_total += (time.perf_counter() - t1) * 1000

                mask_bin = (mask_prob > self.seg_threshold).astype(np.uint8)

                # Place segmentation mask back into full-image coordinates
                full_mask[ey1:ey2, ex1:ex2] = np.maximum(
                    full_mask[ey1:ey2, ex1:ex2], mask_bin * 255
                )

                defect_area_px  = int(mask_bin.sum())
                crop_area_px    = mask_bin.size
                defect_area_pct = (defect_area_px / crop_area_px * 100) if crop_area_px > 0 else 0

                detections.append({
                    'class':           cls_name,
                    'confidence':      round(conf, 4),
                    'bbox':            [x1, y1, x2, y2],
                    'crop_bbox':       [ex1, ey1, ex2, ey2],
                    'mask_prob':       mask_prob,       # crop-relative, float [0,1]
                    'mask_binary':     mask_bin,        # crop-relative, uint8 {0,1}
                    'defect_area_px':  defect_area_px,
                    'defect_area_pct': round(defect_area_pct, 2),
                })

        total_ms = (time.perf_counter() - t0) * 1000

        return {
            'detections': detections,
            'full_mask':  full_mask,
            'timing': {
                'detect_ms':  round(detect_ms, 2),
                'segment_ms': round(segment_ms_total, 2),
                'total_ms':   round(total_ms, 2),
            }
        }


if __name__ == '__main__':
    YOLO_PATH = '/Users/alvi/OpenCV/week2/day14/runs/stage1_detector/weights/best.pt'
    UNET_PATH = '/Users/alvi/OpenCV/output/best_unet.pt'

    pipeline = TwoStageDefectPipeline(YOLO_PATH, UNET_PATH)

    # Quick smoke test on one image
    test_img_path = list(Path('/Users/alvi/OpenCV/data/leather/test/cut').glob('*.png'))[0]
    img = cv2.imread(str(test_img_path))

    result = pipeline.predict(img)

    print(f"\n=== Pipeline test on {test_img_path.name} ===")
    print(f"Detections: {len(result['detections'])}")
    for det in result['detections']:
        print(f"  {det['class']:10s} conf={det['confidence']:.3f} "
              f"bbox={det['bbox']} defect_area={det['defect_area_pct']:.1f}%")
    print(f"\nTiming: detect={result['timing']['detect_ms']:.1f}ms, "
          f"segment={result['timing']['segment_ms']:.1f}ms, "
          f"total={result['timing']['total_ms']:.1f}ms")
    print(f"FPS estimate: {1000/result['timing']['total_ms']:.1f}")