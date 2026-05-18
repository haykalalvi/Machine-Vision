import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load saved calibration
data = np.load('output/camera_calibration.npz')
K, dist = data['K'], data['dist']

# Load a test image
img = cv2.imread('week1/day05/left01.jpg')
h, w = img.shape[:2]

# Get optimal new camera matrix (crops black borders)
K_new, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))

# Undistort
undistorted = cv2.undistort(img, K, dist, None, K_new)

# Crop to ROI
x, y, w_roi, h_roi = roi
undistorted_cropped = undistorted[y:y+h_roi, x:x+w_roi]

# Compare side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax1.set_title(f'Original (distorted)')
ax1.axis('off')
ax2.imshow(cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB))
ax2.set_title('Undistorted (corrected)')
ax2.axis('off')
plt.tight_layout()
plt.savefig('output/undistortion_comparison.jpg', dpi=150)
plt.show()