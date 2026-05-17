import os

import cv2

img = cv2.imread(os.path.join('.','data', 'bear.jpg.webp'))
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# namanA ADAPTIVE threshold, jadi dia bagi jadi 20 itu konstanta pengurangnya
# 33 tu kayak ukuran jendela kecil (kernelnya) berarti 33x33 pixel
thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 33, 20)
cv2.imshow("img", img)
cv2.imshow("img_gray", img_gray)
cv2.imshow("thresh", thresh)
cv2.waitKey(0)