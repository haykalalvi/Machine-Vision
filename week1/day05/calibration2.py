import cv2
import numpy as np
import glob
import os

# Prepare object points for a 9x6 board with 1-unit square size
objp = np.zeros((6 * 9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

obj_points = []  # 3D points in real world
img_points = []  # 2D points in image plane

# img = cv2.imread(os.path.join('.','week1', 'day05', 'left01.jpg'))
images = glob.glob('week1/day05/*.jpg')
print(f"Found {len(images)} calibration images")

successful = 0
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
    
    if ret:
        successful += 1
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        obj_points.append(objp)
        img_points.append(corners_refined)
        print(f"  ✓ {fname.split('/')[-1]}")
    else:
        print(f"  ✗ {fname.split('/')[-1]} — corners not found")

print(f"\n{successful}/{len(images)} images used for calibration")

# Run calibration
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    obj_points, img_points, gray.shape[::-1], None, None
)

print(f"\n=== Camera Matrix (K) ===")
print(K)
print(f"\nfx = {K[0,0]:.2f} pixels")
print(f"fy = {K[1,1]:.2f} pixels")
print(f"cx = {K[0,2]:.2f} pixels (principal point x)")
print(f"cy = {K[1,2]:.2f} pixels (principal point y)")

print(f"\n=== Distortion Coefficients ===")
print(f"k1={dist[0,0]:.4f}, k2={dist[0,1]:.4f}, p1={dist[0,2]:.4f}, p2={dist[0,3]:.4f}, k3={dist[0,4]:.4f}")

print(f"\n=== Reprojection Error ===") # the overall grade of calibrationnya.
# compare calculated pixel to the actual pixel yang berhasil dikalkulasi
print(f"{ret:.4f} pixels  {'✓ Excellent' if ret < 0.5 else '✓ Good' if ret < 1.0 else '⚠ Try more images'}")

# Save calibration results
np.savez('output/camera_calibration.npz', K=K, dist=dist)
print("\nCalibration saved to outputs/camera_calibration.npz")


