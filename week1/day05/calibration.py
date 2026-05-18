import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import os

# Load one image first to verify corner detection works
#img = cv2.imread('data/calib/left01.jpg')
# img = cv2.imread('/Users/alvi/OpenCV/week1/day05/left01.jpg')
img = cv2.imread(os.path.join('.','week1', 'day05', 'left01.jpg'))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# (9,6) = number of INNER corners, not squares
# berarti doi ada 10 squares horizontally dan 7 squares vertically
# Count carefully on your chessboard image
ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

# ret itu boolean yang menunjukkan apakah corner ditemukan
# corners itu koordinat corner yang ditemukan

print(f"Corners found: {ret}")  # must be True

if ret:
    # Refine corner locations to sub-pixel accuracy
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    
    # Draw and display
    img_corners = cv2.drawChessboardCorners(img.copy(), (9, 6), corners_refined, ret)
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(img_corners, cv2.COLOR_BGR2RGB))
    plt.title('Detected chessboard corners')
    plt.axis('off')
    plt.savefig('output/corners_detected.jpg')
    plt.show()