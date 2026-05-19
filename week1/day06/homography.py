import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- CLICK-TO-SELECT VERSION ---
# You will click 4 corners of your object in order:
# top-left, top-right, bottom-right, bottom-left

img = cv2.imread('week1/day06/distort_1.jpg')

# img = cv2.imread('data/document_angled.jpg')
img_display = img.copy()
points = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append([x, y])
        cv2.circle(img_display, (x, y), 8, (0, 255, 0), -1)
        cv2.putText(img_display, str(len(points)), (x+10, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow('Click 4 corners: TL, TR, BR, BL', img_display)
        
        if len(points) == 4:
            print("4 points selected:", points)

cv2.imshow('Click 4 corners: TL, TR, BR, BL', img_display)
cv2.setMouseCallback('Click 4 corners: TL, TR, BR, BL', click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Compute output dimensions (A4 ratio: 210 x 297mm)
output_w, output_h = 600, 850

src_pts = np.float32(points)
dst_pts = np.float32([
    [0,        0       ],   # top-left maps to top-left
    [output_w, 0       ],   # top-right maps to top-right
    [output_w, output_h],   # bottom-right maps to bottom-right
    [0,        output_h]    # bottom-left maps to bottom-left
])

# Compute homography matrix
# H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
H, status = cv2.findHomography(src_pts, dst_pts, 0, 5.0)
print(f"\nHomography matrix H:\n{H}")
print(f"\nInlier ratio: {status.sum()}/{len(status)}")

# Apply the perspective warp
warped = cv2.warpPerspective(img, H, (output_w, output_h))

# Show result
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax1.set_title('Original (perspective distorted)')
ax1.axis('off')
ax2.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
ax2.set_title('Corrected (bird-eye view)')
ax2.axis('off')
plt.tight_layout()
plt.savefig('output/perspective_correction.jpg', dpi=150)
plt.show()