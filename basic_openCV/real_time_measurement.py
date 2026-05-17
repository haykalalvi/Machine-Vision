"""
Real-Time Object Measurement dengan Classical Vision (OpenCV)
============================================================
Menggabungkan Watershed Segmentation + Pixel-to-CM Measurement

Fitur:
- Segmentasi objek otomatis dengan Watershed Algorithm
- Pengukuran dimensi bounding box (lebar & tinggi) setiap objek
- Kalibrasi rasio pixel-to-cm secara interaktif
- Mode kalibrasi: klik 2 titik dengan jarak referensi yang diketahui
- Kontrol keyboard: q=quit, r=reset kalibrasi, s=simpan screenshot

Cara Pakai:
1. Jalankan: python realtime_measurement.py
2. Saat pertama muncul, program langsung mulai mengukur
3. Tekan 'c' untuk masuk mode kalibrasi, lalu klik 2 titik
   dengan panjang referensi yang Anda tahu (misal: 10 cm)
4. Input panjang referensi di terminal, kalibrasi selesai
"""

import cv2 as cv
import numpy as np
import math
import time


# ─── Konfigurasi Default ─────────────────────────────────────────────────────

RATIO_DEFAULT = 10 / 137          # cm per pixel (sesuaikan dengan setup Anda)
MIN_OBJECT_AREA = 500             # area minimum (px²) agar dihitung sebagai objek
CALIBRATION_REF_CM = 10.0        # panjang referensi default saat kalibrasi (cm)


# ─── State Global ─────────────────────────────────────────────────────────────

ratio = RATIO_DEFAULT
calibration_mode = False
calib_points = []
calib_ref_cm = CALIBRATION_REF_CM


# ─── Mouse Callback ───────────────────────────────────────────────────────────

def mouse_callback(event, x, y, flags, param):
    global calib_points, ratio, calibration_mode, calib_ref_cm

    if calibration_mode and event == cv.EVENT_LBUTTONDOWN:
        calib_points.append((x, y))
        print(f"  [Kalibrasi] Titik {len(calib_points)}: ({x}, {y})")

        if len(calib_points) == 2:
            pt1, pt2 = calib_points
            dist_px = math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
            if dist_px > 0:
                ratio = calib_ref_cm / dist_px
                print(f"  [Kalibrasi] Jarak pixel: {dist_px:.1f}px")
                print(f"  [Kalibrasi] Rasio baru : {ratio:.6f} cm/px")
                print(f"  [Kalibrasi] Selesai ✓")
            calib_points = []
            calibration_mode = False


# ─── Watershed Segmentation ───────────────────────────────────────────────────

def segment_objects(frame):
    """
    Mengembalikan list bounding box [(x, y, w, h), ...] dari objek
    yang berhasil tersegmentasi via Watershed.
    """
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Threshold Otsu
    _, thresh = cv.threshold(gray, 0, 255,
                               cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    #thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C , cv.THRESH_BINARY, 93, 20)
    #thresh = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #        cv.THRESH_BINARY,11,10)

    # Noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)

    # Sure background
    sure_bg = cv.dilate(opening, kernel, iterations=3)

    # Sure foreground via distance transform
    dist = cv.distanceTransform(opening, cv.DIST_L2, 5)
    _, sure_fg = cv.threshold(dist, 0.6 * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Unknown region
    unknown = cv.subtract(sure_bg, sure_fg)

    # Marker labelling
    _, markers = cv.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Watershed
    markers = cv.watershed(frame.copy(), markers)

    # Ekstrak bounding box per objek
    bboxes = []
    unique_labels = np.unique(markers)
    for label in unique_labels:
        if label <= 1:          # background & border
            continue
        mask = np.uint8(markers == label) * 255
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL,
                                       cv.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv.contourArea(cnt)
            if area < MIN_OBJECT_AREA:
                continue
            x, y, w, h = cv.boundingRect(cnt)
            bboxes.append((x, y, w, h, cnt))

    return bboxes, markers


# ─── Overlay Visualisasi ──────────────────────────────────────────────────────

def draw_overlay(frame, bboxes, markers, ratio, calibration_mode,
                 calib_points, fps):
    vis = frame.copy()

    # Warnai setiap region watershed
    overlay = np.zeros_like(frame)
    unique_labels = np.unique(markers)
    colors = {}
    for label in unique_labels:
        if label <= 1:
            continue
        if label not in colors:
            np.random.seed(int(label) * 37)
            colors[label] = (
                int(np.random.randint(80, 220)),
                int(np.random.randint(80, 220)),
                int(np.random.randint(80, 220)),
            )
        overlay[markers == label] = colors[label]

    vis = cv.addWeighted(vis, 0.65, overlay, 0.35, 0)

    # Gambar bounding box + ukuran
    for idx, (x, y, w, h, cnt) in enumerate(bboxes):
        w_cm = w * ratio
        h_cm = h * ratio

        # Bounding box
        cv.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 100), 2)

        # Kontur tipis
        cv.drawContours(vis, [cnt], -1, (255, 255, 0), 1)

        # Label ukuran
        label = f"{w_cm:.1f}x{h_cm:.1f}cm"
        lx, ly = x, y - 10 if y > 20 else y + h + 20
        (tw, th), _ = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv.rectangle(vis, (lx - 2, ly - th - 4),
                     (lx + tw + 2, ly + 2), (0, 0, 0), -1)
        cv.putText(vis, label, (lx, ly),
                   cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 100), 1,
                   cv.LINE_AA)

        # Titik pusat
        cx, cy = x + w // 2, y + h // 2
        cv.circle(vis, (cx, cy), 4, (0, 200, 255), -1)

    # ── Titik kalibrasi ──────────────────────────────────────────────────
    if calibration_mode:
        for pt in calib_points:
            cv.circle(vis, pt, 8, (0, 80, 255), -1)
            cv.circle(vis, pt, 10, (255, 255, 255), 2)
        if len(calib_points) == 1:
            cv.putText(vis, "Klik titik ke-2",
                       (calib_points[0][0] + 12, calib_points[0][1] - 12),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 80, 255), 2)

    # ── HUD panel ────────────────────────────────────────────────────────
    h_frame, w_frame = vis.shape[:2]

    # Panel atas
    panel = np.zeros((52, w_frame, 3), dtype=np.uint8)
    panel[:] = (20, 20, 20)
    vis = np.vstack([panel, vis])

    mode_text = "MODE: KALIBRASI" if calibration_mode else "MODE: PENGUKURAN"
    mode_color = (0, 80, 255) if calibration_mode else (0, 255, 100)

    cv.putText(vis, mode_text, (10, 22),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2, cv.LINE_AA)
    cv.putText(vis, f"Objek: {len(bboxes)}",
               (300, 22), cv.FONT_HERSHEY_SIMPLEX, 0.6,
               (200, 200, 200), 1, cv.LINE_AA)
    cv.putText(vis, f"Rasio: {ratio:.5f} cm/px",
               (10, 44), cv.FONT_HERSHEY_SIMPLEX, 0.5,
               (160, 160, 160), 1, cv.LINE_AA)
    cv.putText(vis, f"FPS: {fps:.1f}",
               (w_frame - 100, 22), cv.FONT_HERSHEY_SIMPLEX,
               0.6, (200, 200, 0), 1, cv.LINE_AA)

    # Panel bawah
    bottom = np.zeros((36, w_frame, 3), dtype=np.uint8)
    bottom[:] = (20, 20, 20)
    cv.putText(bottom,
               "q:Keluar  c:Kalibrasi  r:Reset Rasio  s:Screenshot",
               (10, 24), cv.FONT_HERSHEY_SIMPLEX, 0.5,
               (120, 120, 120), 1, cv.LINE_AA)
    vis = np.vstack([vis, bottom])

    return vis


# ─── Main Loop ────────────────────────────────────────────────────────────────

def main():
    global ratio, calibration_mode, calib_points, calib_ref_cm

    print("=" * 55)
    print("  Real-Time Measurement — Classical Vision (OpenCV)")
    print("=" * 55)
    print(f"  Rasio awal  : {ratio:.6f} cm/px")
    print(f"  Kontrol     : q=Keluar | c=Kalibrasi | r=Reset | s=Screenshot")
    print("=" * 55)

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Kamera tidak dapat dibuka!")
        return

    # Resolusi kamera
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

    cv.namedWindow("Real-Time Measurement", cv.WINDOW_NORMAL)
    cv.setMouseCallback("Real-Time Measurement", mouse_callback)

    fps = 0.0
    prev_time = time.time()
    screenshot_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Tidak bisa membaca frame.")
            break

        # FPS
        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        # Segmentasi
        bboxes, markers = segment_objects(frame)

        # Overlay
        vis = draw_overlay(frame, bboxes, markers, ratio,
                           calibration_mode, calib_points, fps)

        cv.imshow("Real-Time Measurement", vis)

        key = cv.waitKey(1) & 0xFF

        # q — keluar
        if key == ord('q'):
            print("\n[Info] Program dihentikan.")
            break

        # c — mulai kalibrasi
        elif key == ord('c'):
            if not calibration_mode:
                try:
                    ref = input("\n[Kalibrasi] Masukkan panjang referensi (cm) "
                                f"[default={calib_ref_cm}]: ").strip()
                    calib_ref_cm = float(ref) if ref else calib_ref_cm
                except ValueError:
                    pass
                calibration_mode = True
                calib_points = []
                print(f"[Kalibrasi] Klik 2 titik dengan jarak {calib_ref_cm} cm di frame.")

        # r — reset rasio ke default
        elif key == ord('r'):
            ratio = RATIO_DEFAULT
            calibration_mode = False
            calib_points = []
            print(f"[Info] Rasio direset ke default: {ratio:.6f} cm/px")

        # s — screenshot
        elif key == ord('s'):
            fname = f"screenshot_{screenshot_count:03d}.png"
            cv.imwrite(fname, vis)
            screenshot_count += 1
            print(f"[Info] Screenshot disimpan: {fname}")

        # Escape — batalkan kalibrasi
        elif key == 27:
            calibration_mode = False
            calib_points = []
            print("[Info] Kalibrasi dibatalkan.")

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()